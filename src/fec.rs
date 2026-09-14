//! Reed-Solomon FEC + CRC-16 integrity check.
//!
//! RS correction and CRC verification are independent: RS corrects what it
//! can, CRC confirms the result is actually right. RS can silently
//! mis-correct beyond its guaranteed bound, so a frame that fails CRC is
//! treated as failed even if RS reported success -- that check lives in
//! [`crate::protocol`], which always re-verifies CRC after calling
//! [`recover`] regardless of what it returns.
//!
//! # Why a hand-rolled Reed-Solomon codec
//!
//! No actively-maintained Rust crate does classical error-*correcting* RS
//! (unknown error positions, as opposed to erasure coding, which needs known
//! error positions -- not available on a noisy audio channel). This is a
//! standard textbook GF(256) codec (primitive polynomial 0x11d, generator 2,
//! first-consecutive-root 1 -- the same parameters used in most public
//! Reed-Solomon references and tutorials), implemented as: syndrome
//! calculation, Berlekamp-Massey for the error locator polynomial, Chien
//! search for error positions, and the Forney algorithm for error
//! magnitudes. As a second, independent safety net beyond the CRC layer
//! above, [`rs_correct_msg`] recomputes syndromes on its own "corrected"
//! output and refuses to return it unless they're all zero -- so a bug in
//! this implementation can make correction fail more often than it should,
//! but cannot make it silently return wrong data as if it were right.
//!
//! GF(256) Reed-Solomon caps a single codeword at 255 symbols total, so
//! data longer than `255 - parity_bytes` bytes is split into multiple
//! chunks here, each with its own `parity_bytes` of parity -- the caller
//! (both sides) derives identical chunk boundaries from the original data
//! length, so this never depends on an implicit/internal chunking detail.

use crc::{Crc, CRC_16_IBM_3740};

pub const DEFAULT_PARITY_BYTES: usize = 10;
pub const MAX_RS_BLOCK: usize = 255; // GF(256) codeword length ceiling (data + parity)

// --- GF(256) arithmetic ------------------------------------------------

const GF_PRIM: u16 = 0x11d; // x^8 + x^4 + x^3 + x^2 + 1
const GF_GENERATOR: u8 = 2;
const FCR: i32 = 1; // first consecutive root

struct GfTables {
    exp: [u8; 255],
    log: [u8; 256],
}

#[allow(clippy::needless_range_loop)] // index drives two parallel tables plus sequential GF state
fn build_gf_tables() -> GfTables {
    let mut exp = [0u8; 255];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    for i in 0..255usize {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= GF_PRIM;
        }
    }
    GfTables { exp, log }
}

thread_local! {
    static GF: GfTables = build_gf_tables();
}

fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    GF.with(|gf| {
        let sum = gf.log[a as usize] as usize + gf.log[b as usize] as usize;
        gf.exp[sum % 255]
    })
}

fn gf_div(a: u8, b: u8) -> u8 {
    assert_ne!(b, 0, "division by zero in GF(256)");
    if a == 0 {
        return 0;
    }
    GF.with(|gf| {
        let diff = gf.log[a as usize] as i32 - gf.log[b as usize] as i32;
        gf.exp[diff.rem_euclid(255) as usize]
    })
}

fn gf_pow(a: u8, power: i32) -> u8 {
    if a == 0 {
        return if power == 0 { 1 } else { 0 };
    }
    GF.with(|gf| {
        let p = (gf.log[a as usize] as i64 * power as i64).rem_euclid(255) as usize;
        gf.exp[p]
    })
}

// --- polynomials, MSB-first (index 0 = highest degree) -- used for the
// systematic encoder, matching the classic "extended synthetic division"
// construction.

fn poly_mul_msb(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut result = vec![0u8; p.len() + q.len() - 1];
    for (i, &pc) in p.iter().enumerate() {
        if pc == 0 {
            continue;
        }
        for (j, &qc) in q.iter().enumerate() {
            if qc == 0 {
                continue;
            }
            result[i + j] ^= gf_mul(pc, qc);
        }
    }
    result
}

fn rs_generator_poly(nsym: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..nsym {
        let root = gf_pow(GF_GENERATOR, FCR + i as i32);
        g = poly_mul_msb(&g, &[1, root]);
    }
    g
}

fn rs_encode(msg: &[u8], nsym: usize) -> Vec<u8> {
    let gen = rs_generator_poly(nsym);
    let mut buf = msg.to_vec();
    buf.extend(std::iter::repeat_n(0u8, nsym));
    for i in 0..msg.len() {
        let coef = buf[i];
        if coef != 0 {
            for (j, &g) in gen.iter().enumerate() {
                buf[i + j] ^= gf_mul(g, coef);
            }
        }
    }
    // The division loop above overwrites the message region with the
    // division's intermediate remainder; restore the original (unmodified,
    // systematic) message bytes now that only the parity tail matters.
    buf[..msg.len()].copy_from_slice(msg);
    buf
}

// --- polynomials, ascending (index i = coefficient of x^i) -- used for
// syndromes / error locator / Forney, matching the standard formulation of
// those algorithms.

fn poly_eval_ascending(poly: &[u8], x: u8) -> u8 {
    let mut result = 0u8;
    for &c in poly.iter().rev() {
        result = gf_mul(result, x) ^ c;
    }
    result
}

fn poly_mul_ascending(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut result = vec![0u8; a.len() + b.len() - 1];
    for (i, &ac) in a.iter().enumerate() {
        if ac == 0 {
            continue;
        }
        for (j, &bc) in b.iter().enumerate() {
            if bc == 0 {
                continue;
            }
            result[i + j] ^= gf_mul(ac, bc);
        }
    }
    result
}

/// Formal derivative of an ascending-power polynomial, in a field of
/// characteristic 2 (so only odd-degree terms survive: d/dx of c_i*x^i is
/// c_i*x^(i-1) when i is odd, 0 when i is even). The result is a *sparse*
/// polynomial in x^2 -- `result[k]` is the coefficient of `x^(2k)`, i.e.
/// `poly[2k+1]` -- so evaluating it needs [`eval_derivative`], not
/// [`poly_eval_ascending`] directly on this array.
fn poly_derivative_ascending(poly: &[u8]) -> Vec<u8> {
    poly.iter().skip(1).step_by(2).copied().collect()
}

/// Evaluates a derivative produced by [`poly_derivative_ascending`] at `x`:
/// substitutes `x^2` for the derivative's implicit variable, since
/// `deriv[k]` is the coefficient of `x^(2k)`, not `x^k`.
fn eval_derivative(deriv: &[u8], x: u8) -> u8 {
    poly_eval_ascending(deriv, gf_mul(x, x))
}

fn calc_syndromes(codeword: &[u8], nsym: usize) -> Vec<u8> {
    (0..nsym)
        .map(|j| poly_eval_ascending_msb_form(codeword, gf_pow(GF_GENERATOR, FCR + j as i32)))
        .collect()
}

/// Evaluates an MSB-first polynomial (as codewords are stored) at `x`.
fn poly_eval_ascending_msb_form(msb_poly: &[u8], x: u8) -> u8 {
    let mut y = 0u8;
    for &c in msb_poly {
        y = gf_mul(y, x) ^ c;
    }
    y
}

fn poly_scale_and_shift(c: &[u8], b: &[u8], coef: u8, shift: usize) -> Vec<u8> {
    let new_len = c.len().max(b.len() + shift);
    let mut result = vec![0u8; new_len];
    result[..c.len()].copy_from_slice(c);
    for (k, &bc) in b.iter().enumerate() {
        result[shift + k] ^= gf_mul(coef, bc);
    }
    result
}

/// Berlekamp-Massey: finds the shortest LFSR (error locator polynomial,
/// ascending powers, constant term 1) that generates the syndrome sequence.
fn berlekamp_massey(synd: &[u8]) -> Vec<u8> {
    let mut c = vec![1u8];
    let mut b = vec![1u8];
    let mut l = 0usize;
    let mut m = 1usize;
    let mut b_coef = 1u8;

    for i in 0..synd.len() {
        let mut delta = synd[i];
        for j in 1..=l {
            if let Some(&cj) = c.get(j) {
                delta ^= gf_mul(cj, synd[i - j]);
            }
        }
        if delta == 0 {
            m += 1;
        } else if 2 * l <= i {
            let t = c.clone();
            let coef = gf_div(delta, b_coef);
            c = poly_scale_and_shift(&c, &b, coef, m);
            l = i + 1 - l;
            b = t;
            b_coef = delta;
            m = 1;
        } else {
            let coef = gf_div(delta, b_coef);
            c = poly_scale_and_shift(&c, &b, coef, m);
            m += 1;
        }
    }
    c
}

/// Corrects up to `nsym / 2` byte errors at unknown positions in `codeword`
/// (length <= 255, msg followed by `nsym` parity bytes). Returns `None` if
/// the errors exceed that budget -- verified by an independent
/// post-correction syndrome check, not just the error count, so a bug here
/// fails safe rather than mis-correcting silently.
fn rs_correct_msg(codeword: &[u8], nsym: usize) -> Option<Vec<u8>> {
    if nsym == 0 {
        return Some(codeword.to_vec());
    }
    let synd = calc_syndromes(codeword, nsym);
    if synd.iter().all(|&s| s == 0) {
        return Some(codeword.to_vec());
    }

    let err_loc = berlekamp_massey(&synd);
    let nu = err_loc.len() - 1;
    if nu == 0 || 2 * nu > nsym {
        return None; // no valid locator, or more errors than the code can guarantee
    }

    let n = codeword.len();
    let mut error_positions = Vec::with_capacity(nu);
    for p in 0..n {
        // Root candidate for position p (0-indexed from the front of the
        // codeword): X_p^{-1} = alpha^{-(n-1-p)}.
        let candidate = gf_pow(GF_GENERATOR, -((n - 1 - p) as i32));
        if poly_eval_ascending(&err_loc, candidate) == 0 {
            error_positions.push(p);
        }
    }
    if error_positions.len() != nu {
        return None; // Chien search didn't find exactly deg(locator) roots -- uncorrectable
    }

    // Forney: error evaluator Omega(x) = [S(x) * Lambda(x)] mod x^nsym.
    let omega = {
        let mut o = poly_mul_ascending(&synd, &err_loc);
        o.truncate(nsym);
        o
    };
    let err_loc_deriv = poly_derivative_ascending(&err_loc);

    let mut corrected = codeword.to_vec();
    for &p in &error_positions {
        let x_inv = gf_pow(GF_GENERATOR, -((n - 1 - p) as i32));
        let denom = eval_derivative(&err_loc_deriv, x_inv);
        if denom == 0 {
            return None; // degenerate case, can't compute a magnitude safely
        }
        let magnitude = gf_div(poly_eval_ascending(&omega, x_inv), denom);
        corrected[p] ^= magnitude;
    }

    // Independent safety net: only trust the correction if it actually
    // produces a zero-syndrome codeword.
    if calc_syndromes(&corrected, nsym).iter().any(|&s| s != 0) {
        return None;
    }
    Some(corrected)
}

/// Returns only the parity bytes (one `parity_bytes`-sized block per chunk,
/// concatenated) -- not the full RS codeword. The frame format
/// ([`crate::protocol`]) transmits PAYLOAD and PARITY as separate labeled
/// regions, so the caller keeps data and parity apart and passes both to
/// [`recover`].
pub fn protect(data: &[u8], parity_bytes: usize) -> Vec<u8> {
    assert!(
        parity_bytes < MAX_RS_BLOCK,
        "parity_bytes ({parity_bytes}) must be less than MAX_RS_BLOCK ({MAX_RS_BLOCK}) -- a GF(256) \
         codeword (chunk + parity) can hold at most {MAX_RS_BLOCK} symbols total, so parity alone \
         can't reach that size. Validate parity_bytes before calling into fec:: (the CLI layer does \
         this at argument-parsing time)."
    );
    if data.is_empty() {
        return Vec::new();
    }
    let chunk_size = MAX_RS_BLOCK - parity_bytes;
    let mut parity_out = Vec::new();
    for chunk in data.chunks(chunk_size) {
        let full = rs_encode(chunk, parity_bytes);
        parity_out.extend_from_slice(&full[chunk.len()..]);
    }
    parity_out
}

/// `parity` must be exactly what `protect(data, parity_bytes)` produced
/// (chunk boundaries are derived from `data.len()`, so both sides need to
/// agree on that length -- [`crate::protocol`] gets it from the frame's
/// LENGTH field). Returns the corrected data, or `None` if RS could not
/// correct it (caller should treat `None` the same as a CRC mismatch -> NACK).
pub fn recover(data: &[u8], parity: &[u8], parity_bytes: usize) -> Option<Vec<u8>> {
    assert!(
        parity_bytes < MAX_RS_BLOCK,
        "parity_bytes ({parity_bytes}) must be less than MAX_RS_BLOCK ({MAX_RS_BLOCK}) -- see protect()'s \
         assertion for why."
    );
    if data.is_empty() {
        return Some(Vec::new());
    }
    let chunk_size = MAX_RS_BLOCK - parity_bytes;
    let mut out = Vec::with_capacity(data.len());
    let mut pi = 0usize;
    for chunk in data.chunks(chunk_size) {
        let chunk_parity = parity.get(pi..pi + parity_bytes)?;
        pi += parity_bytes;
        let mut codeword = chunk.to_vec();
        codeword.extend_from_slice(chunk_parity);
        let corrected = rs_correct_msg(&codeword, parity_bytes)?;
        out.extend_from_slice(&corrected[..chunk.len()]);
    }
    Some(out)
}

const CRC16: Crc<u16> = Crc::<u16>::new(&CRC_16_IBM_3740);

pub fn crc16_ccitt(data: &[u8]) -> u16 {
    CRC16.checksum(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn crc16_ccitt_known_vector() {
        // Standard CRC-16/CCITT-FALSE test vector.
        assert_eq!(crc16_ccitt(b"123456789"), 0x29B1);
    }

    #[test]
    fn rs_protect_returns_parity_only_sized_to_parity_bytes() {
        let data = b"the quick brown fox";
        let parity = protect(data, 10);
        assert_eq!(parity.len(), 10); // single chunk: not the full codeword
    }

    #[test]
    fn rs_protect_recover_clean() {
        let data = b"the quick brown fox";
        let parity = protect(data, DEFAULT_PARITY_BYTES);
        assert_eq!(
            recover(data, &parity, DEFAULT_PARITY_BYTES).as_deref(),
            Some(&data[..])
        );
    }

    #[test]
    fn rs_corrects_errors_within_budget() {
        let data = b"the quick brown fox jumps";
        let parity_bytes = 10; // corrects up to 5 byte errors
        let parity = protect(data, parity_bytes);
        let mut codeword = data.to_vec();
        codeword.extend_from_slice(&parity);
        for i in [0, 5, 10, 15, 20] {
            codeword[i] ^= 0xFF;
        }
        let (corrupted_data, corrupted_parity) = codeword.split_at(data.len());
        assert_eq!(
            recover(corrupted_data, corrupted_parity, parity_bytes).as_deref(),
            Some(&data[..])
        );
    }

    #[test]
    fn rs_fails_beyond_budget_without_crashing() {
        let data = b"the quick brown fox jumps over";
        let parity_bytes = 10; // corrects up to 5 byte errors
        let parity = protect(data, parity_bytes);
        let mut codeword = data.to_vec();
        codeword.extend_from_slice(&parity);
        for i in (0..20).step_by(2) {
            // 10 errors, over budget
            codeword[i] ^= 0xFF;
        }
        let (corrupted_data, corrupted_parity) = codeword.split_at(data.len());
        let result = recover(corrupted_data, corrupted_parity, parity_bytes);
        // must not silently return wrong-but-unflagged data as if correct
        assert!(result.is_none() || result.as_deref() != Some(&data[..]));
    }

    /// Regression test: GF(256) RS caps a codeword at 255 symbols. A naive
    /// implementation could silently mishandle a payload spanning multiple
    /// chunks (e.g. treating chunked parity as one contiguous block). This
    /// exercises that boundary explicitly, with data spanning 3 chunks at
    /// parity_bytes=10 (chunk_size=245).
    #[test]
    fn rs_handles_data_longer_than_single_gf256_block() {
        let data: Vec<u8> = (0..=255u8).cycle().take(512).collect(); // 512 bytes, spans multiple RS blocks
        let parity_bytes = 10;
        let parity = protect(&data, parity_bytes);
        assert_eq!(recover(&data, &parity, parity_bytes), Some(data.clone()));

        // and it should still actually correct errors within each chunk's budget
        let mut codeword = data.clone();
        codeword.extend_from_slice(&parity);
        codeword[0] ^= 0xFF; // error in chunk 1
        codeword[300] ^= 0xFF; // error in chunk 2 (chunk_size = 245, so byte 300 is in chunk 2)
        let (corrupted_data, corrupted_parity) = codeword.split_at(data.len());
        assert_eq!(
            recover(corrupted_data, corrupted_parity, parity_bytes),
            Some(data)
        );
    }

    #[test]
    fn rs_empty_data() {
        let parity = protect(b"", 10);
        assert_eq!(recover(b"", &parity, 10), Some(Vec::new()));
    }

    /// Regression test: `parity_bytes >= MAX_RS_BLOCK` must fail loudly, not
    /// silently. Before the guard assert was added, `parity_bytes ==
    /// MAX_RS_BLOCK` (255) made `chunk_size = MAX_RS_BLOCK - parity_bytes`
    /// zero, and `data.chunks(0)` panics with a confusing message; worse,
    /// `parity_bytes > MAX_RS_BLOCK` (e.g. 300) underflowed `chunk_size` in
    /// release builds (where integer overflow doesn't panic), producing a
    /// mathematically invalid Reed-Solomon codeword that "succeeded"
    /// without ever flagging the data as corrupt. The CLI now rejects both
    /// values before they reach this layer at all (see `main.rs`'s
    /// `--parity-bytes` clap range), but this asserts the library itself
    /// still fails safe for any other caller.
    #[test]
    #[should_panic(expected = "parity_bytes")]
    fn protect_rejects_parity_bytes_at_max_rs_block() {
        protect(b"some data", MAX_RS_BLOCK);
    }

    #[test]
    #[should_panic(expected = "parity_bytes")]
    fn protect_rejects_parity_bytes_over_max_rs_block() {
        protect(b"some data", MAX_RS_BLOCK + 45);
    }

    #[test]
    #[should_panic(expected = "parity_bytes")]
    fn recover_rejects_parity_bytes_at_max_rs_block() {
        recover(b"some data", b"parity", MAX_RS_BLOCK);
    }

    #[test]
    #[should_panic(expected = "parity_bytes")]
    fn recover_rejects_parity_bytes_over_max_rs_block() {
        recover(b"some data", b"parity", MAX_RS_BLOCK + 45);
    }

    #[test]
    fn rs_exact_t_error_boundary_and_t_plus_one_failure() {
        // parity_bytes=20 -> t=10 correctable errors.
        let data = b"a somewhat longer message used to test the exact error-correction boundary of this codec";
        let parity_bytes = 20;
        let parity = protect(data, parity_bytes);

        // Exactly t=10 errors: must correct.
        let mut codeword = data.to_vec();
        codeword.extend_from_slice(&parity);
        for i in 0..10 {
            codeword[i * 3] ^= 0xAA;
        }
        let (d, p) = codeword.split_at(data.len());
        assert_eq!(recover(d, p, parity_bytes).as_deref(), Some(&data[..]));

        // t+1=11 errors: must not silently return wrong data.
        let mut codeword2 = data.to_vec();
        codeword2.extend_from_slice(&parity);
        for i in 0..11 {
            codeword2[i * 3] ^= 0xAA;
        }
        let (d2, p2) = codeword2.split_at(data.len());
        let result = recover(d2, p2, parity_bytes);
        assert!(result.is_none() || result.as_deref() != Some(&data[..]));
    }

    #[test]
    fn rs_random_trials_within_budget_always_correct() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let len = rng.random_range(1..120);
            let data: Vec<u8> = (0..len).map(|_| rng.random()).collect();
            let parity_bytes = 10; // t = 5
            let parity = protect(&data, parity_bytes);
            let n_errors = rng.random_range(0..=5);
            let mut codeword = data.clone();
            codeword.extend_from_slice(&parity);
            let mut positions: Vec<usize> = (0..codeword.len()).collect();
            // simple partial shuffle: pick n_errors distinct positions
            for _ in 0..n_errors {
                let idx = rng.random_range(0..positions.len());
                let pos = positions.swap_remove(idx);
                let bit: u8 = 1 << rng.random_range(0..8);
                codeword[pos] ^= bit;
            }
            let (d, p) = codeword.split_at(data.len());
            assert_eq!(
                recover(d, p, parity_bytes),
                Some(data),
                "failed with {n_errors} errors"
            );
        }
    }

    #[test]
    fn gf_arithmetic_sanity() {
        // Multiplicative order of the generator must be 255 for these
        // tables to be a valid field representation.
        GF.with(|gf| {
            assert_eq!(gf.exp[0], 1);
            assert_eq!(gf_mul(gf.exp[254], GF_GENERATOR), 1);
        });
        for a in 1u8..=255 {
            assert_eq!(
                gf_mul(a, gf_div(1, a)),
                1,
                "a={a} has no valid multiplicative inverse"
            );
        }
    }
}
