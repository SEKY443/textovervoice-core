//! End-to-end encryption: X25519 key exchange + ChaCha20-Poly1305 AEAD.
//!
//! Uses `x25519-dalek` (dalek-cryptography) and `chacha20poly1305`
//! (RustCrypto/AEADs) -- hand-rolling any crypto primitive here would be
//! reckless regardless of how simple it looks; unlike [`crate::fec`]'s
//! Reed-Solomon codec, these are exactly the primitives it's worth pulling
//! a vetted, widely-used library for.
//!
//! X25519 is a key-AGREEMENT scheme, not "encrypt with public key, decrypt
//! with private key" like RSA: both sides run ECDH using (their own private
//! key + the peer's public key) and derive the SAME shared session key.
//! This is the modern standard approach (same family as Signal Protocol,
//! SSH, TLS 1.3) -- it doesn't encrypt data directly, and doesn't need to:
//! it exists to set up a session key once, cheaply, which a fast symmetric
//! cipher then uses per message. This matters a lot on a ~15-20 char/sec
//! channel, where a raw RSA-style ciphertext (190+ bytes minimum regardless
//! of plaintext length) would dominate transmission time;
//! ChaCha20-Poly1305's fixed 28-byte overhead (12-byte nonce + 16-byte auth
//! tag) per message is the only recurring crypto cost here.
//!
//! The ECDH shared secret is NOT used directly as the cipher key -- it's
//! passed through HKDF-SHA256 first (standard practice: raw ECDH output
//! isn't uniformly random enough to use directly as a symmetric key).
//!
//! Caveat for demo use: there is no public-key verification here -- no
//! PKI, no out-of-band fingerprint check like Signal's safety numbers. A
//! key exchange carried out live over an open acoustic channel is, in
//! principle, vulnerable to a man-in-the-middle who intercepts and
//! substitutes their own public key. This is fine for demonstrating
//! encrypted-in-transit messaging; it is not a claim of protection against
//! an active attacker without adding that verification step separately.

use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce};
use hkdf::Hkdf;
use sha2::Sha256;
use x25519_dalek::{PublicKey, StaticSecret};

pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16; // produced by ChaCha20Poly1305 as part of its ciphertext output

const SESSION_INFO: &[u8] = b"textovervoice-session";

pub fn generate_keypair() -> (StaticSecret, PublicKey) {
    let private_key = StaticSecret::random();
    let public_key = PublicKey::from(&private_key);
    (private_key, public_key)
}

pub fn serialize_public_key(pub_key: &PublicKey) -> [u8; 32] {
    pub_key.to_bytes()
}

pub fn load_public_key(data: [u8; 32]) -> PublicKey {
    PublicKey::from(data)
}

pub fn serialize_private_key(priv_key: &StaticSecret) -> [u8; 32] {
    priv_key.to_bytes()
}

pub fn load_private_key(data: [u8; 32]) -> StaticSecret {
    StaticSecret::from(data)
}

/// ECDH shared secret -> HKDF-SHA256 -> 32-byte symmetric key. Both sides
/// land on the identical key: Alice with (her private, Bob's public), Bob
/// with (his private, Alice's public).
pub fn derive_session_key(my_private: &StaticSecret, their_public: &PublicKey) -> [u8; 32] {
    let shared_secret = my_private.diffie_hellman(their_public);
    let hk = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut okm = [0u8; 32];
    hk.expand(SESSION_INFO, &mut okm)
        .expect("32 is a valid HKDF-SHA256 output length");
    okm
}

pub fn encrypt(session_key: &[u8; 32], plaintext: &[u8]) -> Vec<u8> {
    let mut nonce_bytes = [0u8; NONCE_LEN];
    getrandom::fill(&mut nonce_bytes).expect("OS CSPRNG failure");
    let cipher = ChaCha20Poly1305::new(session_key.into());
    let nonce = Nonce::try_from(nonce_bytes.as_slice()).expect("NONCE_LEN matches Nonce's size");
    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .expect("ChaCha20Poly1305 encryption cannot fail for well-formed input");
    let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

/// Returns the plaintext, or `None` if authentication failed (tampered
/// ciphertext, or simply the wrong key -- ChaCha20-Poly1305 can't tell
/// those apart, which is the correct behavior: it shouldn't leak which one
/// happened).
pub fn decrypt(session_key: &[u8; 32], blob: &[u8]) -> Option<Vec<u8>> {
    if blob.len() < NONCE_LEN + TAG_LEN {
        return None;
    }
    let (nonce, ciphertext) = blob.split_at(NONCE_LEN);
    let nonce = Nonce::try_from(nonce).expect("split at NONCE_LEN");
    let cipher = ChaCha20Poly1305::new(session_key.into());
    cipher.decrypt(&nonce, ciphertext).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecdh_both_sides_derive_same_session_key() {
        let (alice_priv, alice_pub) = generate_keypair();
        let (bob_priv, bob_pub) = generate_keypair();

        let alice_key = derive_session_key(&alice_priv, &bob_pub);
        let bob_key = derive_session_key(&bob_priv, &alice_pub);
        assert_eq!(alice_key, bob_key);
        assert_eq!(alice_key.len(), 32);
    }

    #[test]
    fn different_keypairs_derive_different_session_keys() {
        let (alice_priv, _alice_pub) = generate_keypair();
        let (_bob_priv, bob_pub) = generate_keypair();
        let (_eve_priv, eve_pub) = generate_keypair();

        let alice_bob_key = derive_session_key(&alice_priv, &bob_pub);
        let alice_eve_key = derive_session_key(&alice_priv, &eve_pub);
        assert_ne!(alice_bob_key, alice_eve_key);
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let (priv_key, pub_key) = generate_keypair();
        let key = derive_session_key(&priv_key, &pub_key); // self-exchange is fine for this test
        let plaintext = "Hello, encrypted world! 你好".as_bytes();

        let blob = encrypt(&key, plaintext);
        assert_eq!(decrypt(&key, &blob).as_deref(), Some(plaintext));
    }

    #[test]
    fn tampered_ciphertext_fails_auth() {
        let (priv_key, pub_key) = generate_keypair();
        let key = derive_session_key(&priv_key, &pub_key);
        let mut blob = encrypt(&key, b"secret message");
        let last = blob.len() - 1;
        blob[last] ^= 0xFF;
        assert_eq!(decrypt(&key, &blob), None);
    }

    #[test]
    fn wrong_key_fails_to_decrypt() {
        let (alice_priv, alice_pub) = generate_keypair();
        let (_bob_priv, bob_pub) = generate_keypair();
        let (eve_priv, _eve_pub) = generate_keypair();

        let alice_key = derive_session_key(&alice_priv, &bob_pub);
        let eve_key = derive_session_key(&eve_priv, &alice_pub); // not the shared key

        let blob = encrypt(&alice_key, b"for bob's eyes only");
        assert_eq!(decrypt(&eve_key, &blob), None);
    }

    #[test]
    fn public_key_serialization_roundtrip() {
        let (_priv_key, pub_key) = generate_keypair();
        let data = serialize_public_key(&pub_key);
        assert_eq!(data.len(), 32); // X25519 public keys are 32 raw bytes
        let restored = load_public_key(data);
        assert_eq!(serialize_public_key(&restored), data);
    }

    #[test]
    fn private_key_serialization_roundtrip() {
        let (priv_key, _pub_key) = generate_keypair();
        let data = serialize_private_key(&priv_key);
        assert_eq!(data.len(), 32);
        let restored = load_private_key(data);
        // verify it's functionally the same key via a session-key derivation
        let (_other_priv, other_pub) = generate_keypair();
        let k1 = derive_session_key(&priv_key, &other_pub);
        let k2 = derive_session_key(&restored, &other_pub);
        assert_eq!(k1, k2);
    }

    #[test]
    fn each_encryption_uses_a_fresh_nonce() {
        let (priv_key, pub_key) = generate_keypair();
        let key = derive_session_key(&priv_key, &pub_key);
        let blob1 = encrypt(&key, b"same message");
        let blob2 = encrypt(&key, b"same message");
        assert_ne!(blob1, blob2); // different nonce each time, even for identical plaintext
    }
}
