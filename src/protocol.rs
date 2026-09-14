//! Frame assembly/parsing: ties charset + fec + framing (+ optional crypto)
//! together into the wire format. Two formats exist -- see [`FrameFormat`]
//! -- and [`parse_frame`]/[`frame_wire_length`] auto-detect which one a
//! received frame uses (no out-of-band negotiation needed): a bare
//! [`codes::FEC_SCHEME_ID_LO`] immediately after SOF marks
//! [`FrameFormat::ProtectedHeader`]; anything else there is a
//! [`FrameFormat::Legacy`] frame's (always-stuffed, never-bare) DEST_ID
//! byte, and `stuff()`'s "never emit a reserved code bare" invariant makes
//! that distinction unambiguous even under channel corruption of the
//! DEST_ID byte's *value*.
//!
//! ```text
//! Legacy:          SOF | DEST_ID | SRC_ID | FLAGS | SEQ | LENGTH(2B) |
//!                  PAYLOAD | PARITY_START | RS PARITY (stuffed) |
//!                  PARITY_END | CRC_HI | CRC(2B, stuffed) | CRC_LO
//!
//! ProtectedHeader: SOF | SCHEME_ID_LO | HEADER(6B, stuffed) |
//!                  HEADER PARITY (stuffed) | PAYLOAD |
//!                  RS PARITY (stuffed) | CRC(2B, stuffed)
//! ```
//!
//! DEST_ID, SRC_ID, FLAGS, and SEQ are opt-in features, all default to "off"
//! (`DEST_ID=BROADCAST_ID`, `SRC_ID=UNKNOWN_SRC_ID`, `FLAGS=0`/plaintext/
//! single-frame, `SEQ=0`) -- addressing, sender identification, encryption,
//! and multi-frame messages are strict additions, not requirements, so
//! plain single-frame broadcast usage is unaffected.
//!
//! SRC_ID identifies who sent a frame -- distinct from DEST_ID (who it's
//! addressed to). It exists for multi-party use (e.g. several people on one
//! shared voice channel): with only DEST_ID, a receiver of a broadcast frame
//! has no way to tell which of several possible senders it came from. Like
//! DEST_ID it's opt-in plaintext metadata, not authenticated -- an attacker
//! controlling the channel can forge it freely, same caveat that already
//! applies to DEST_ID.
//!
//! SEQ + `FLAG_MORE_FRAMES` exist because a single preamble at the start of
//! a long transmission can't correct for timing drift that accumulates
//! *during* the transmission (measured on real AMR-NB at 4.75kbps against
//! the full GPLv3 text: a hard failure "cliff" partway through, not
//! scattered noise) -- see [`crate::message`] for the splitting/reassembly
//! logic built on top of this.
//!
//! FEC and CRC protect the exact wire-form payload bytes (post-stuffing, as
//! they travel over the channel) -- not a resolved/unescaped form. The
//! receiver can't safely unescape before FEC correction anyway, since a
//! corrupted escape sequence would misparse. So the flow is: collect raw
//! wire codes for the payload region -> FEC-correct -> CRC-verify -> only
//! then hand the corrected wire codes to charset/dictionary decoding
//! (plaintext) or crypto decryption, each of which does its own
//! escape/flag resolution.
//!
//! # Why two formats
//!
//! [`FrameFormat::Legacy`]'s DEST_ID/FLAGS/SEQ/LENGTH header -- and, worse,
//! its PARITY_START/PARITY_END/CRC_HI/CRC_LO *marker bytes themselves* --
//! carry no redundancy at all. A single corrupted marker byte anywhere in
//! the payload region makes the whole frame structurally unparseable
//! (`find_bare_flag` can't locate the boundary it's scanning for),
//! completely independent of how much FEC protects the payload's own
//! *content*. This is not a theoretical concern: re-validating this port
//! against real AMR-NB at its worst bitrate (4.75kbps) reproduced exactly
//! this failure ("PARITY_START marker not found") on a message whose
//! payload corruption was well within the configured FEC budget --
//! raising `--parity-bytes` did nothing for it, because the actual damage
//! was to an unprotected marker byte, not the payload.
//!
//! A second, subtler consequence of the same zero-redundancy header:
//! DEST_ID/SRC_ID/SEQ/FLAGS aren't covered by the CRC either (it only
//! covers the payload text), so noise confined to header bytes -- quite
//! plausible by chance with a handful of scattered bit flips, no marker
//! byte involved at all -- is invisible to every check `Legacy` runs. A
//! 5000-trial random-corruption fuzz sweep measured this at ~7% of trials
//! reporting `ok` with *correct* text but a *wrong* dest_id/src_id/seq (see
//! `random_corruption_sweep_never_reports_wrong_fields` in this module's
//! tests). Harmless if a caller doesn't rely on those fields; a real
//! problem for addressing, multi-frame reassembly, or (the reason this was
//! actually found) SRC_ID-based sender attribution in group chat.
//!
//! [`FrameFormat::ProtectedHeader`] fixes both problems at the root: the
//! 6-byte header is itself wrapped in a small RS codeword (reusing the
//! exact same [`fec::protect`]/[`fec::recover`] machinery the payload
//! already uses -- no new math, no new failure modes to reason about).
//! Once the header (and therefore LENGTH) is verified, every other
//! region's position is computed directly from it -- no scanning, so
//! PARITY_START/PARITY_END/CRC_HI/CRC_LO become unnecessary and are
//! dropped entirely, and header fields get the same detect-or-correct
//! guarantee the payload already had (the same fuzz sweep above measured
//! 0/5000 wrong-field reports for this format). The result is both more
//! robust *and* structurally simpler than the format it replaces, at the
//! cost of 1 extra structural byte/frame (`SCHEME_ID_LO` plus
//! `HEADER_PARITY_BYTES`, minus the 4 dropped marker bytes) -- plus,
//! occasionally, one more: `HEADER_PARITY_BYTES` are effectively random RS
//! output, and a byte that happens to collide with a reserved code costs 1
//! extra wire byte to escape, same as any other parity/CRC byte in either
//! format. It's the default; `Legacy` is kept for anyone who can tolerate
//! both risks above (e.g. a low-noise digital channel, or use that doesn't
//! depend on header field integrity) and wants that byte back, or wants to
//! interoperate with encoders that predate this addition.

use crate::framing::{stuff, unstuff_bytes, Token, TokenReader};
use crate::{charset, codes, crypto, dictionary, fec};

pub const BROADCAST_ID: u8 = 255; // dest_id value meaning "for everyone" -- the default
pub const UNKNOWN_SRC_ID: u8 = 255; // src_id value meaning "sender not specified" -- the default
pub const FLAG_ENCRYPTED: u8 = 0b0000_0001;
pub const FLAG_MORE_FRAMES: u8 = 0b0000_0010; // message continues in a subsequent frame

/// Number of RS parity bytes protecting the 6-byte header in
/// [`FrameFormat::ProtectedHeader`] frames -- corrects up to 6 corrupted
/// header bytes (`t = parity/2`). Originally fixed at 4 (t=2) on the
/// reasoning that the header's *size* never varies, so there's no
/// per-message tuning question the way there is for payload
/// `--parity-bytes`. Real acoustic phone-channel testing (TOVChat, a
/// browser port of this format) showed that reasoning misses the axis
/// that actually matters: channel *quality*, not header size. Calibrate
/// mode lets a caller raise payload `parity_bytes` to survive a noisy
/// channel, but the header had no equivalent -- so on exactly the noisy
/// channels calibrate exists for, the header (stuck at t=2) became the
/// bottleneck and failed before the payload's own (now well-protected)
/// FEC was ever exercised. Raised to t=6; the header is only 6 bytes, so
/// even a generous parity budget costs a handful of wire bytes per frame.
const HEADER_PARITY_BYTES: usize = 12;
const HEADER_LEN: usize = 6; // DEST_ID(1) + SRC_ID(1) + FLAGS(1) + SEQ(1) + LENGTH(2)

/// RS parity bytes protecting a [`NackFrame`]'s 5-byte header -- same
/// budget as [`HEADER_PARITY_BYTES`] (t=6 correction), applied to a
/// slightly shorter block. A resend request is exactly the kind of short,
/// high-value control message worth protecting at least as well as an
/// ordinary frame's header: if it can't survive the channel, the receiver
/// silently gets no resend instead of a clean failure it could retry.
const NACK_PARITY_BYTES: usize = 12;
/// DEST_ID(1) + SRC_ID(1) + TARGET_ID(3) -- see [`NackFrame`].
const NACK_HEADER_LEN: usize = 5;

/// Which wire format [`build_frame`] emits and [`parse_frame`] expects to
/// auto-detect. See this module's docs for the full rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameFormat {
    /// Original format: unprotected header, PARITY_START/PARITY_END/CRC_HI/
    /// CRC_LO markers scanned for at decode time. Because DEST_ID/SRC_ID/
    /// SEQ/FLAGS carry no redundancy and the CRC only covers the payload
    /// text (not the header), channel noise confined to header bytes is
    /// invisible to every check this format runs -- measured at ~7% of
    /// trials in a random-corruption fuzz sweep (see
    /// `random_corruption_sweep_never_reports_wrong_fields` in this
    /// module's tests) reporting `ok` with a *correct* text but a *wrong*
    /// dest_id/src_id/seq. Not a concern if you don't rely on those
    /// fields (e.g. plain unaddressed single-sender use); a real one if
    /// you do (addressing, multi-frame reassembly, or SRC_ID-based sender
    /// attribution in group chat).
    Legacy,
    /// RS-protected header, no marker scanning. The default: strictly more
    /// robust against real-channel corruption, at a small (usually 1-byte)
    /// overhead per frame.
    #[default]
    ProtectedHeader,
}

/// The header fields common to both wire formats, once fully read --
/// threaded through the rest of parsing/failure-reporting as one value
/// instead of several separate positional arguments. `more_frames` is
/// deliberately not a separate field: it's always exactly
/// `flags & FLAG_MORE_FRAMES != 0`, so storing it alongside `flags` would
/// just be two copies of the same fact that could drift apart.
#[derive(Debug, Clone, Copy)]
struct FrameHeader {
    dest_id: u8,
    src_id: u8,
    seq: u8,
    flags: u8,
}

impl FrameHeader {
    fn more_frames(&self) -> bool {
        self.flags & FLAG_MORE_FRAMES != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    pub ok: bool,
    pub text: Option<String>,
    pub reason: String,
    pub dest_id: Option<u8>,
    pub src_id: Option<u8>,
    pub seq: Option<u8>,
    pub more_frames: Option<bool>,
}

impl ParseResult {
    fn fail(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            text: None,
            reason: reason.into(),
            dest_id: None,
            src_id: None,
            seq: None,
            more_frames: None,
        }
    }

    fn fail_with(reason: impl Into<String>, header: FrameHeader) -> Self {
        Self {
            ok: false,
            text: None,
            reason: reason.into(),
            dest_id: Some(header.dest_id),
            src_id: Some(header.src_id),
            seq: Some(header.seq),
            more_frames: Some(header.more_frames()),
        }
    }

    fn not_for_me(dest_id: u8, src_id: u8) -> Self {
        Self {
            ok: false,
            text: None,
            reason: format!("not addressed to me (dest_id={dest_id})"),
            dest_id: Some(dest_id),
            src_id: Some(src_id),
            seq: None,
            more_frames: None,
        }
    }
}

/// A resend request: "please retransmit the message identified by
/// `target_id`". Distinct from an ordinary data frame -- recognized right
/// after SOF (via [`codes::FEC_SCHEME_ID_HI`], reserved for exactly this:
/// "a future third frame format") without running dictionary/charset
/// decode, FEC over a full payload, or CRC -- so a listener can act on it
/// cheaply and a corrupted one fails obviously rather than being
/// misinterpreted as a malformed data frame.
///
/// `target_id` identifies which message to resend. It's opaque to this
/// module -- callers are expected to tag every frame of a message they
/// send with the same 3-byte id (at the payload/text layer, e.g. as a
/// short prefix), so that any single successfully-decoded frame reveals
/// which message it belongs to even if other frames of that message were
/// lost. 3 bytes matches the id size already used for this purpose
/// elsewhere (see CLI-TextOverVoice's `chat.rs` `random_msg_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NackFrame {
    pub dest_id: u8,
    pub src_id: u8,
    pub target_id: [u8; 3],
}

/// Builds a resend-request frame. Always succeeds -- a 5-byte header can't
/// overflow anything the way a payload's LENGTH field can.
pub fn build_nack_frame(nack: &NackFrame) -> Vec<u8> {
    let header_data = [
        nack.dest_id,
        nack.src_id,
        nack.target_id[0],
        nack.target_id[1],
        nack.target_id[2],
    ];
    let header_parity = fec::protect(&header_data, NACK_PARITY_BYTES);

    let mut frame = vec![codes::FEC_SOF, codes::FEC_SCHEME_ID_HI];
    frame.extend(stuff(&header_data));
    frame.extend(stuff(&header_parity));
    frame
}

/// Attempts to read a [`NackFrame`] starting at `received[0]`. Returns
/// `None` both when `received` isn't a NACK frame at all (wrong dispatch
/// byte -- most commonly because it's an ordinary data frame instead) and
/// when it is one but the header proved uncorrectable -- a caller scanning
/// for resend requests only cares whether a valid one for it arrived, not
/// why a malformed one didn't parse, unlike [`parse_frame`]'s richer
/// [`ParseResult`] (built for a human/log-facing "why did this fail").
pub fn parse_nack_frame(received: &[u8]) -> Option<NackFrame> {
    let mut reader = TokenReader::new(received);
    match reader.read().ok()? {
        Token::Flag(v) if v == codes::FEC_SOF => {}
        _ => return None,
    }
    match reader.read().ok()? {
        Token::Flag(v) if v == codes::FEC_SCHEME_ID_HI => {}
        _ => return None,
    }
    let header = read_nack_header(&mut reader)?;
    Some(NackFrame {
        dest_id: header[0],
        src_id: header[1],
        target_id: [header[2], header[3], header[4]],
    })
}

/// Shared by [`parse_nack_frame`] and [`frame_wire_length`]'s NACK branch:
/// reads the header+parity block as logical tokens (same rationale as
/// [`read_protected_header`] for treating a broken escape-pair structurally
/// the same as a corrupted data byte) and RS-corrects it.
fn read_nack_header(reader: &mut TokenReader) -> Option<[u8; NACK_HEADER_LEN]> {
    let mut header_and_parity = Vec::with_capacity(NACK_HEADER_LEN + NACK_PARITY_BYTES);
    for _ in 0..(NACK_HEADER_LEN + NACK_PARITY_BYTES) {
        match reader.read().ok()? {
            Token::Data(v) | Token::Flag(v) => header_and_parity.push(v),
        }
    }
    let (header_data, header_parity) = header_and_parity.split_at(NACK_HEADER_LEN);
    let corrected = fec::recover(header_data, header_parity, NACK_PARITY_BYTES)?;
    corrected.try_into().ok()
}

/// Options accepted by [`build_frame`]. `seq`/`more_frames` are normally set
/// by [`crate::message::build_message`], not by a direct caller.
pub struct BuildOptions<'a> {
    pub parity_bytes: usize,
    pub use_dictionary: bool,
    pub dest_id: u8,
    pub src_id: u8,
    pub session_key: Option<&'a [u8; 32]>,
    pub seq: u8,
    pub more_frames: bool,
    pub frame_format: FrameFormat,
}

impl Default for BuildOptions<'_> {
    fn default() -> Self {
        Self {
            parity_bytes: fec::DEFAULT_PARITY_BYTES,
            use_dictionary: true,
            dest_id: BROADCAST_ID,
            src_id: UNKNOWN_SRC_ID,
            session_key: None,
            seq: 0,
            more_frames: false,
            frame_format: FrameFormat::default(),
        }
    }
}

/// Largest payload the wire format can address: LENGTH is a 2-byte field
/// (see [`build_frame`]), so a payload longer than this can't be declared
/// correctly -- `payload_bytes.len() as u16` would silently truncate/wrap
/// instead of erroring, producing a frame with a LENGTH field that doesn't
/// match its actual payload (an undecodable frame that `build_frame`
/// nonetheless "successfully" returns). This is realistically reachable:
/// a large `--max-frame-chars` on non-dictionary-compressible text (each
/// char can cost several wire bytes once UTF-8/escaping is accounted for)
/// crosses this well before `u16::MAX` characters.
pub const MAX_PAYLOAD_BYTES: usize = u16::MAX as usize;

/// Encodes `text` per `opts` (plaintext charset/dictionary, or encrypted)
/// and applies `FLAG_MORE_FRAMES` -- the part identical between both wire
/// formats. Returns `(payload_bytes, flags)`.
fn encode_payload(text: &str, opts: &BuildOptions) -> (Vec<u8>, u8) {
    let (payload_codes, flags_base) = if let Some(session_key) = opts.session_key {
        let ciphertext = crypto::encrypt(session_key, text.as_bytes());
        (stuff(&ciphertext), FLAG_ENCRYPTED)
    } else {
        let codes = if opts.use_dictionary {
            dictionary::encode_text(text)
        } else {
            charset::encode_text(text)
        };
        (codes, 0u8)
    };
    let flags = if opts.more_frames {
        flags_base | FLAG_MORE_FRAMES
    } else {
        flags_base
    };
    (payload_codes, flags)
}

/// Number of RS parity bytes [`fec::protect`] produces for a payload of
/// `data_len` bytes at `parity_bytes` per chunk -- lets a caller compute
/// the parity region's size without re-deriving `fec`'s chunking formula
/// by hand. `0` for empty data, matching `fec::protect`'s own behavior.
fn total_parity_len(data_len: usize, parity_bytes: usize) -> usize {
    if data_len == 0 {
        return 0;
    }
    let chunk_size = fec::MAX_RS_BLOCK - parity_bytes;
    data_len.div_ceil(chunk_size) * parity_bytes
}

/// `dest_id`: optional addressee, 0-254 for a specific recipient or
/// `BROADCAST_ID` (default) for everyone -- addressing is opt-in.
///
/// `session_key`: if given (see [`crypto::derive_session_key`]), the
/// payload is ChaCha20-Poly1305-encrypted instead of charset/dictionary-
/// encoded text -- also opt-in, off by default. When set, `use_dictionary`
/// is ignored (encrypted bytes aren't text, there's nothing to compress).
///
/// Returns `None` if the encoded payload would exceed [`MAX_PAYLOAD_BYTES`]
/// -- a data-dependent outcome (depends on the actual text/dictionary hits/
/// encryption overhead), not a fixed-config caller mistake, so it's
/// reported the same way [`fec::recover`] reports "couldn't, for reasons
/// intrinsic to this input" rather than panicking.
pub fn build_frame(text: &str, opts: &BuildOptions) -> Option<Vec<u8>> {
    match opts.frame_format {
        FrameFormat::Legacy => build_frame_legacy(text, opts),
        FrameFormat::ProtectedHeader => build_frame_protected(text, opts),
    }
}

fn build_frame_legacy(text: &str, opts: &BuildOptions) -> Option<Vec<u8>> {
    let (payload_bytes, flags) = encode_payload(text, opts);
    if payload_bytes.len() > MAX_PAYLOAD_BYTES {
        return None;
    }

    let parity = fec::protect(&payload_bytes, opts.parity_bytes);
    let crc = fec::crc16_ccitt(&payload_bytes);

    let mut frame = vec![codes::FEC_SOF];
    frame.extend(stuff(&[opts.dest_id]));
    frame.extend(stuff(&[opts.src_id]));
    frame.extend(stuff(&[flags]));
    frame.extend(stuff(&[opts.seq]));
    frame.extend(stuff(&(payload_bytes.len() as u16).to_be_bytes()));
    frame.extend(&payload_bytes);
    frame.push(codes::FEC_PARITY_START);
    frame.extend(stuff(&parity));
    frame.push(codes::FEC_PARITY_END);
    frame.push(codes::CRC_MARKER_HI);
    frame.extend(stuff(&crc.to_be_bytes()));
    frame.push(codes::CRC_MARKER_LO);
    Some(frame)
}

fn build_frame_protected(text: &str, opts: &BuildOptions) -> Option<Vec<u8>> {
    let (payload_bytes, flags) = encode_payload(text, opts);
    if payload_bytes.len() > MAX_PAYLOAD_BYTES {
        return None;
    }

    let parity = fec::protect(&payload_bytes, opts.parity_bytes);
    let crc = fec::crc16_ccitt(&payload_bytes);

    let len_bytes = (payload_bytes.len() as u16).to_be_bytes();
    let header_data = [
        opts.dest_id,
        opts.src_id,
        flags,
        opts.seq,
        len_bytes[0],
        len_bytes[1],
    ];
    let header_parity = fec::protect(&header_data, HEADER_PARITY_BYTES);

    let mut frame = vec![codes::FEC_SOF, codes::FEC_SCHEME_ID_LO];
    frame.extend(stuff(&header_data));
    frame.extend(stuff(&header_parity));
    frame.extend(&payload_bytes);
    frame.extend(stuff(&parity));
    frame.extend(stuff(&crc.to_be_bytes()));
    Some(frame)
}

/// Raw-code boundary scan: skips escape pairs (2 codes) without resolving
/// them, since we need the boundary index, not the value. A bare reserved
/// code that isn't `target` (e.g. a UTF-8 flag inside the payload) is just
/// payload content at this layer and is skipped as ordinary data. Returns
/// `None` if `target` never appears bare. [`FrameFormat::Legacy`] only.
fn find_bare_flag(wire: &[u8], start: usize, target: u8) -> Option<usize> {
    let mut i = start;
    while i < wire.len() {
        let c = wire[i];
        if c == codes::ESCAPE {
            i += 2;
            continue;
        }
        if c == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

macro_rules! expect_flag {
    ($reader:expr, $name:expr, $value:expr) => {
        match $reader.read() {
            Ok(Token::Flag(v)) if v == $value => {}
            Ok(other) => {
                return ParseResult::fail(format!(
                    "expected {} ({}), got {:?}",
                    $name, $value, other
                ))
            }
            Err(_) => return ParseResult::fail(format!("expected {} ({}), got EOF", $name, $value)),
        }
    };
}

macro_rules! read_data_byte {
    ($reader:expr) => {
        match $reader.read() {
            Ok(Token::Data(v)) => v,
            Ok(Token::Flag(v)) => {
                return ParseResult::fail(format!("expected data byte, got flag={v}"))
            }
            Err(_) => return ParseResult::fail("unexpected end of frame"),
        }
    };
}

/// Scans one frame starting at `start` and returns the total number of wire
/// codes it occupies (SOF through the frame's final byte, inclusive), or
/// `None` if the structure can't even be walked. Exists for multi-frame
/// scanning (live audio capture, file decoding): a caller demodulating a
/// generous, over-sized audio window needs to know exactly where THIS
/// frame's real content ends, so the next preamble search starts precisely
/// there -- not a guessed window that might contain zero, one, or several
/// subsequent preambles. `parity_bytes` must match what the frame was
/// (expected to be) built with -- only used by the `ProtectedHeader` path,
/// which computes the parity region's size directly instead of scanning
/// for it.
pub fn frame_wire_length(wire: &[u8], start: usize, parity_bytes: usize) -> Option<usize> {
    let mut reader = TokenReader::new(&wire[start..]);
    match reader.read().ok()? {
        Token::Flag(v) if v == codes::FEC_SOF => {}
        _ => return None,
    }

    let saved = reader.pos();
    match reader.read() {
        Ok(Token::Flag(v)) if v == codes::FEC_SCHEME_ID_LO => {
            frame_wire_length_protected(&wire[start..], &mut reader, parity_bytes)
                .map(|len| start + len)
        }
        Ok(Token::Flag(v)) if v == codes::FEC_SCHEME_ID_HI => {
            read_nack_header(&mut reader)?;
            Some(start + reader.pos())
        }
        _ => {
            reader.set_pos(saved);
            frame_wire_length_legacy(&wire[start..], &mut reader).map(|len| start + len)
        }
    }
}

fn frame_wire_length_legacy(wire: &[u8], reader: &mut TokenReader) -> Option<usize> {
    reader.read().ok()?; // dest_id
    reader.read().ok()?; // src_id
    reader.read().ok()?; // flags
    reader.read().ok()?; // seq
    reader.read().ok()?; // length byte 1
    reader.read().ok()?; // length byte 2

    let payload_start = reader.pos();
    let idx = find_bare_flag(wire, payload_start, codes::FEC_PARITY_START)?;
    reader.set_pos(idx);
    reader.read().ok()?; // PARITY_START

    let parity_start2 = reader.pos();
    let idx2 = find_bare_flag(wire, parity_start2, codes::FEC_PARITY_END)?;
    reader.set_pos(idx2);

    reader.read().ok()?; // PARITY_END
    reader.read().ok()?; // CRC_HI
    reader.read().ok()?; // crc byte 1
    reader.read().ok()?; // crc byte 2
    reader.read().ok()?; // CRC_LO

    Some(reader.pos())
}

/// Distinguishes *why* [`read_protected_header`] couldn't produce a header
/// -- found live against a real acoustic channel: a header read from a
/// live-captured buffer that's still growing (the rest of the transmission
/// hasn't arrived yet) hits the exact same code path as a header that
/// arrived complete but failed RS correction, and collapsing both into one
/// "uncorrectable" outcome makes them indistinguishable to a caller. A live
/// poll loop needs that distinction: "ran out of audio" means wait and
/// retry once more has arrived; "RS correction failed" means the header
/// that *did* arrive is genuinely corrupt, safe to report immediately.
/// Every other region (payload/parity/CRC) already reports its own
/// "unexpected end of frame..." for this same ambiguity; the header just
/// hadn't been given one.
enum HeaderRead {
    Ok([u8; HEADER_LEN]),
    Truncated,
    Uncorrectable,
}

/// Reads the `HEADER_LEN + HEADER_PARITY_BYTES`-byte protected header block
/// as logical tokens (resolving escapes, since header byte *values* may
/// legitimately need it) and RS-corrects it.
///
/// A `Token::Flag` here means channel corruption broke an escape pair's
/// *structure* (turned what should have been data into what looks like a
/// bare reserved code) rather than just its *value* -- but that's still
/// just one wrong byte as far as RS is concerned. Using the flag's raw
/// value as the (corrupted) data byte and letting `fec::recover` attempt
/// correction, instead of immediately giving up, matches
/// [`unstuff_bytes`]'s same philosophy for exactly this situation: failing
/// fast here would make this format's header *less* resilient to this one
/// corruption pattern than the plain byte-stuffing helper already is.
fn read_protected_header(reader: &mut TokenReader) -> HeaderRead {
    let mut header_and_parity = Vec::with_capacity(HEADER_LEN + HEADER_PARITY_BYTES);
    for _ in 0..(HEADER_LEN + HEADER_PARITY_BYTES) {
        match reader.read() {
            Ok(Token::Data(v) | Token::Flag(v)) => header_and_parity.push(v),
            Err(_) => return HeaderRead::Truncated,
        }
    }
    let (header_data, header_parity) = header_and_parity.split_at(HEADER_LEN);
    let Some(corrected) = fec::recover(header_data, header_parity, HEADER_PARITY_BYTES) else {
        return HeaderRead::Uncorrectable;
    };
    match corrected.try_into() {
        Ok(bytes) => HeaderRead::Ok(bytes),
        Err(_) => HeaderRead::Uncorrectable,
    }
}

fn frame_wire_length_protected(
    wire: &[u8],
    reader: &mut TokenReader,
    parity_bytes: usize,
) -> Option<usize> {
    let HeaderRead::Ok(header) = read_protected_header(reader) else {
        return None;
    };
    let declared_len = u16::from_be_bytes([header[4], header[5]]) as usize;

    let payload_start = reader.pos();
    if payload_start + declared_len > wire.len() {
        return None;
    }
    reader.set_pos(payload_start + declared_len);

    for _ in 0..total_parity_len(declared_len, parity_bytes) {
        reader.read().ok()?;
    }
    reader.read().ok()?; // crc byte 1
    reader.read().ok()?; // crc byte 2

    Some(reader.pos())
}

/// `use_dictionary` must match what [`build_frame`] used for plaintext
/// frames, or dictionary hits will be misread.
///
/// `my_id`: if given, a frame not addressed to this id (and not
/// `BROADCAST_ID`) is rejected immediately after reading DEST_ID -- before
/// FEC/CRC even run -- both for the "only the intended receiver responds"
/// effect and as a real efficiency win.
///
/// `session_key`: required to decrypt a frame with `FLAG_ENCRYPTED` set;
/// without it such a frame is reported as failed, not silently ignored.
///
/// Auto-detects [`FrameFormat`] from the wire -- see this module's docs.
pub fn parse_frame(
    received: &[u8],
    parity_bytes: usize,
    use_dictionary: bool,
    my_id: Option<u8>,
    session_key: Option<&[u8; 32]>,
) -> ParseResult {
    let mut reader = TokenReader::new(received);
    expect_flag!(reader, "SOF", codes::FEC_SOF);

    let saved = reader.pos();
    match reader.read() {
        Ok(Token::Flag(v)) if v == codes::FEC_SCHEME_ID_LO => parse_frame_protected(
            received,
            &mut reader,
            parity_bytes,
            use_dictionary,
            my_id,
            session_key,
        ),
        _ => {
            reader.set_pos(saved);
            parse_frame_legacy(
                received,
                &mut reader,
                parity_bytes,
                use_dictionary,
                my_id,
                session_key,
            )
        }
    }
}

fn parse_frame_legacy(
    received: &[u8],
    reader: &mut TokenReader,
    parity_bytes: usize,
    use_dictionary: bool,
    my_id: Option<u8>,
    session_key: Option<&[u8; 32]>,
) -> ParseResult {
    let dest_id = read_data_byte!(reader);
    let src_id = read_data_byte!(reader);

    if let Some(id) = my_id {
        if dest_id != BROADCAST_ID && dest_id != id {
            return ParseResult::not_for_me(dest_id, src_id);
        }
    }

    let flags = read_data_byte!(reader);
    let seq = read_data_byte!(reader);
    let header = FrameHeader {
        dest_id,
        src_id,
        seq,
        flags,
    };
    let len_hi = read_data_byte!(reader);
    let len_lo = read_data_byte!(reader);
    let declared_len = u16::from_be_bytes([len_hi, len_lo]) as usize;

    let payload_start = reader.pos();
    let Some(parity_start_idx) = find_bare_flag(received, payload_start, codes::FEC_PARITY_START)
    else {
        return ParseResult::fail_with("PARITY_START marker not found", header);
    };
    let payload_wire = &received[payload_start..parity_start_idx];

    reader.set_pos(parity_start_idx);
    expect_flag!(reader, "PARITY_START", codes::FEC_PARITY_START);

    let parity_start2 = reader.pos();
    let Some(parity_end_idx) = find_bare_flag(received, parity_start2, codes::FEC_PARITY_END)
    else {
        return ParseResult::fail_with("PARITY_END marker not found", header);
    };
    let parity_wire = &received[parity_start2..parity_end_idx];

    reader.set_pos(parity_end_idx);
    expect_flag!(reader, "PARITY_END", codes::FEC_PARITY_END);
    expect_flag!(reader, "CRC_HI", codes::CRC_MARKER_HI);
    let crc_hi = read_data_byte!(reader);
    let crc_lo = read_data_byte!(reader);
    expect_flag!(reader, "CRC_LO", codes::CRC_MARKER_LO);
    let received_crc = u16::from_be_bytes([crc_hi, crc_lo]);

    if payload_wire.len() != declared_len {
        return ParseResult::fail_with(
            format!(
                "length mismatch: declared {declared_len}, got {}",
                payload_wire.len()
            ),
            header,
        );
    }

    // parity_wire is stuffed on the wire; unstuff it back to raw parity
    // bytes before handing it to the RS decoder.
    let Ok(parity_bytes_val) = unstuff_bytes(parity_wire) else {
        return ParseResult::fail_with("malformed parity region", header);
    };

    finish_parse(
        payload_wire,
        &parity_bytes_val,
        received_crc,
        parity_bytes,
        use_dictionary,
        session_key,
        header,
    )
}

fn parse_frame_protected(
    received: &[u8],
    reader: &mut TokenReader,
    parity_bytes: usize,
    use_dictionary: bool,
    my_id: Option<u8>,
    session_key: Option<&[u8; 32]>,
) -> ParseResult {
    let header_bytes = match read_protected_header(reader) {
        HeaderRead::Ok(bytes) => bytes,
        HeaderRead::Truncated => {
            return ParseResult::fail("unexpected end of frame reading protected header")
        }
        HeaderRead::Uncorrectable => return ParseResult::fail("protected header FEC uncorrectable"),
    };
    let header = FrameHeader {
        dest_id: header_bytes[0],
        src_id: header_bytes[1],
        flags: header_bytes[2],
        seq: header_bytes[3],
    };
    let declared_len = u16::from_be_bytes([header_bytes[4], header_bytes[5]]) as usize;

    if let Some(id) = my_id {
        if header.dest_id != BROADCAST_ID && header.dest_id != id {
            return ParseResult::not_for_me(header.dest_id, header.src_id);
        }
    }

    let payload_start = reader.pos();
    if payload_start + declared_len > received.len() {
        return ParseResult::fail_with("payload extends past end of received data", header);
    }
    let payload_wire = &received[payload_start..payload_start + declared_len];
    reader.set_pos(payload_start + declared_len);

    // A `Token::Flag` in either region below means an escape pair's
    // *structure* broke under corruption, not just its value -- same
    // situation `read_protected_header` handles above, and for the same
    // reason: using the flag's raw value as the (corrupted) byte and
    // letting FEC/CRC catch it downstream is strictly more robust than
    // failing fast on a structural technicality, matching
    // [`unstuff_bytes`]'s philosophy instead of diverging from it.
    let mut parity_bytes_val = Vec::with_capacity(total_parity_len(declared_len, parity_bytes));
    for _ in 0..total_parity_len(declared_len, parity_bytes) {
        match reader.read() {
            Ok(Token::Data(v) | Token::Flag(v)) => parity_bytes_val.push(v),
            Err(_) => {
                return ParseResult::fail_with("unexpected end of frame reading parity", header)
            }
        }
    }

    let mut crc_raw = [0u8; 2];
    for slot in &mut crc_raw {
        match reader.read() {
            Ok(Token::Data(v) | Token::Flag(v)) => *slot = v,
            Err(_) => return ParseResult::fail_with("unexpected end of frame reading CRC", header),
        }
    }
    let received_crc = u16::from_be_bytes(crc_raw);

    finish_parse(
        payload_wire,
        &parity_bytes_val,
        received_crc,
        parity_bytes,
        use_dictionary,
        session_key,
        header,
    )
}

/// Shared tail of both parse paths once the header, payload_wire, parity,
/// and CRC have all been extracted: FEC-correct, CRC-verify, then decode
/// plaintext or decrypt.
fn finish_parse(
    payload_wire: &[u8],
    parity_bytes_val: &[u8],
    received_crc: u16,
    parity_bytes: usize,
    use_dictionary: bool,
    session_key: Option<&[u8; 32]>,
    header: FrameHeader,
) -> ParseResult {
    let Some(corrected) = fec::recover(payload_wire, parity_bytes_val, parity_bytes) else {
        return ParseResult::fail_with("FEC uncorrectable", header);
    };

    if fec::crc16_ccitt(&corrected) != received_crc {
        return ParseResult::fail_with("CRC mismatch after FEC", header);
    }

    let flags_encrypted = header.flags & FLAG_ENCRYPTED != 0;
    let text = if flags_encrypted {
        let Some(session_key) = session_key else {
            return ParseResult::fail_with(
                "frame is encrypted but no session key was provided",
                header,
            );
        };
        let Ok(raw_ciphertext) = unstuff_bytes(&corrected) else {
            return ParseResult::fail_with("malformed encrypted payload", header);
        };
        match crypto::decrypt(session_key, &raw_ciphertext) {
            Some(plaintext) => String::from_utf8_lossy(&plaintext).into_owned(),
            None => {
                return ParseResult::fail_with("decryption failed (wrong key or tampered)", header);
            }
        }
    } else if use_dictionary {
        dictionary::decode_codes(&corrected)
    } else {
        charset::decode_codes(&corrected)
    };

    ParseResult {
        ok: true,
        text: Some(text),
        reason: String::new(),
        dest_id: Some(header.dest_id),
        src_id: Some(header.src_id),
        seq: Some(header.seq),
        more_frames: Some(header.more_frames()),
    }
}

/// A best-effort, non-authoritative snapshot of a
/// [`FrameFormat::ProtectedHeader`] frame's payload text, decoded directly
/// from the raw wire bytes captured so far -- before FEC correction,
/// before CRC verification, possibly before the whole frame has even
/// arrived. Exists to drive a live "as it's heard" decode readout instead
/// of only ever showing text once [`parse_frame`] has fully verified it.
///
/// Safe to call on every poll against a still-growing buffer, at the same
/// position each time: [`preview_frame_protected`] returns `None` if the
/// header hasn't resolved yet, and once it has, this only ever reads
/// however much of the declared payload is currently present -- never
/// errors, never panics, just reports less.
///
/// This is deliberately *not* authoritative. [`fec::protect`]'s RS coding
/// is systematic (data bytes travel as-is; parity is appended separately,
/// never mixed into the data), so on a clean channel these raw bytes
/// already equal what [`parse_frame`] will eventually confirm -- but
/// "usually" isn't "always": a genuine transmission error can make this
/// preview briefly show the wrong character until the real, FEC/CRC-
/// verified result either confirms or corrects it. Showing that
/// correction is the caller's job; this function only ever reports what
/// the raw signal currently looks like, nothing more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePreview {
    pub dest_id: u8,
    pub src_id: u8,
    pub seq: u8,
    pub more_frames: bool,
    /// Total payload length this frame declares, in wire bytes (stuffed,
    /// pre-FEC) -- lets a caller judge how much of the payload `text` was
    /// actually built from (`bytes_seen`/`declared_len`), e.g. to show
    /// progress.
    pub declared_len: usize,
    /// How many of those `declared_len` wire bytes have actually arrived
    /// in the buffer so far.
    pub bytes_seen: usize,
    /// Best-effort decode of `bytes_seen` raw payload bytes.
    pub text: String,
}

/// Builds a [`FramePreview`] from `received` (starting at SOF, same
/// convention as [`parse_frame`]/[`frame_wire_length`]). `use_dictionary`
/// must match what the sender used, same requirement as [`parse_frame`].
///
/// Returns `None` if: this isn't a [`FrameFormat::ProtectedHeader`] frame
/// (nothing safe to preview off an unprotected [`FrameFormat::Legacy`]
/// header -- see this module's docs on why that format's header carries no
/// redundancy at all); the header hasn't resolved (not enough audio yet,
/// or genuinely uncorrectable -- a caller can't tell which from `None`
/// alone, same as [`read_protected_header`]'s own ambiguity, but a preview
/// has no failure-reason display to report it to anyway); `my_id` is given
/// and this frame isn't addressed to it (never preview someone else's
/// message); or the frame is encrypted ([`crypto::decrypt`] needs the
/// complete, FEC-corrected ciphertext to produce anything at all, so
/// there's nothing meaningful to preview before the frame is fully in
/// hand).
pub fn preview_frame_protected(
    received: &[u8],
    use_dictionary: bool,
    my_id: Option<u8>,
) -> Option<FramePreview> {
    let mut reader = TokenReader::new(received);
    match reader.read().ok()? {
        Token::Flag(v) if v == codes::FEC_SOF => {}
        _ => return None,
    }
    match reader.read().ok()? {
        Token::Flag(v) if v == codes::FEC_SCHEME_ID_LO => {}
        _ => return None,
    }
    let HeaderRead::Ok(header_bytes) = read_protected_header(&mut reader) else {
        return None;
    };
    let dest_id = header_bytes[0];
    let src_id = header_bytes[1];
    let flags = header_bytes[2];
    let seq = header_bytes[3];
    let declared_len = u16::from_be_bytes([header_bytes[4], header_bytes[5]]) as usize;

    if let Some(id) = my_id {
        if dest_id != BROADCAST_ID && dest_id != id {
            return None;
        }
    }
    if flags & FLAG_ENCRYPTED != 0 {
        return None;
    }

    // Guaranteed <= received.len(): reader.pos() only ever advances past
    // tokens it actually read from within `received`, and header resolved
    // successfully above.
    let payload_start = reader.pos();
    let bytes_seen = received.len().saturating_sub(payload_start).min(declared_len);
    let payload_wire = &received[payload_start..payload_start + bytes_seen];

    let text = if use_dictionary {
        dictionary::decode_codes(payload_wire)
    } else {
        charset::decode_codes(payload_wire)
    };

    Some(FramePreview {
        dest_id,
        src_id,
        seq,
        more_frames: flags & FLAG_MORE_FRAMES != 0,
        declared_len,
        bytes_seen,
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(text: &str) -> Vec<u8> {
        build_frame(text, &BuildOptions::default()).unwrap()
    }

    /// Regression check for the module doc's overhead claim:
    /// `ProtectedHeader` is *larger* than `Legacy` per frame (not smaller as
    /// an earlier version of this doc comment claimed, based on an
    /// arithmetic error rather than a measurement) -- structurally by
    /// exactly 1 byte, though occasionally more: `HEADER_PARITY_BYTES` are
    /// effectively random RS output, and a byte that happens to collide
    /// with a reserved code costs 1 extra wire byte to escape -- the same
    /// probabilistic tax the payload's own RS parity/CRC bytes already pay
    /// in both formats, just now also possible on this format's header
    /// parity specifically.
    #[test]
    fn protected_header_overhead_is_at_least_one_byte_more_than_legacy() {
        for text in ["hi", "a longer message with more payload bytes in it"] {
            let legacy = build_with_format(text, FrameFormat::Legacy);
            let protected = build_with_format(text, FrameFormat::ProtectedHeader);
            assert!(
                protected.len() > legacy.len(),
                "text={text:?} legacy={} protected={}",
                legacy.len(),
                protected.len()
            );
        }
    }

    fn build_with_format(text: &str, frame_format: FrameFormat) -> Vec<u8> {
        build_frame(
            text,
            &BuildOptions {
                frame_format,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn parse(received: &[u8]) -> ParseResult {
        parse_frame(received, fec::DEFAULT_PARITY_BYTES, true, None, None)
    }

    #[test]
    fn clean_roundtrip_ascii() {
        let frame = build("Hello, World!");
        let result = parse(&frame);
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some("Hello, World!"));
    }

    #[test]
    fn clean_roundtrip_cjk_and_emoji() {
        let text = "文字轉聲音穿越電話通道 \u{1F600}";
        let frame = build(text);
        let result = parse(&frame);
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some(text));
    }

    #[test]
    fn survives_corruption_within_fec_budget() {
        let text = "The quick brown fox jumps over the lazy dog";
        let mut corrupted = build_frame(
            text,
            &BuildOptions {
                parity_bytes: 10,
                ..Default::default()
            },
        )
        .unwrap();
        // Flip a few bytes near (but not at) the very end -- well within
        // the payload/parity region regardless of which frame format's
        // (variably-sized, stuffing-dependent) header precedes it, and
        // well within the 5-byte-error budget for 10 parity bytes. Skips
        // the last few bytes to avoid the CRC field itself.
        for offset in [6, 12, 18] {
            let i = corrupted.len().saturating_sub(offset);
            corrupted[i] ^= 0x01;
        }
        let result = parse_frame(&corrupted, 10, true, None, None);
        assert!(result.ok, "{result:?}");
        assert_eq!(result.text.as_deref(), Some(text));
    }

    #[test]
    fn detects_uncorrectable_corruption_instead_of_returning_garbage() {
        let text = "The quick brown fox jumps over the lazy dog";
        let mut corrupted = build_frame(
            text,
            &BuildOptions {
                parity_bytes: 10,
                ..Default::default()
            },
        )
        .unwrap();
        // Heavily corrupt a wide trailing span (payload+parity+CRC),
        // format-agnostic for the same reason as the test above.
        for offset in (6..46).step_by(2) {
            let i = corrupted.len().saturating_sub(offset);
            corrupted[i] ^= 0xFF;
        }
        let result = parse_frame(&corrupted, 10, true, None, None);
        // Must not silently hand back wrong text as if it were correct.
        assert!(!result.ok || result.text.as_deref() == Some(text));
    }

    /// Fuzz-style sweep aimed specifically at the newest code (the 6-byte
    /// header now carrying SRC_ID, and its `FrameHeader`-based plumbing)
    /// rather than re-covering what the fixed-pattern tests above already
    /// do: random dest_id/src_id/seq, random bit-flip corruption positions
    /// and counts, across both wire formats. The property under test is
    /// the same one enforced everywhere else in this codebase -- a frame
    /// is allowed to fail cleanly, but if it reports `ok`, every field it
    /// reports (dest_id, src_id, seq, text) must be exactly right. A weaker
    /// "recovers whenever corruption count <= budget" claim would actually
    /// be *wrong* to assert here: corrupting a wire byte that happens to be
    /// half of an escape pair can turn a data byte into a structural flag
    /// token instead of a content error RS can correct (see
    /// `codes::FEC_SCHEME_ID_LO`'s doc comment on this residual risk) --
    /// that must show up as a clean failure, not a crash or wrong answer,
    /// which is exactly what this test checks for.
    /// Fuzz-style sweep aimed specifically at the newest code (the 6-byte
    /// header now carrying SRC_ID, and its `FrameHeader`-based plumbing)
    /// rather than re-covering what the fixed-pattern tests above already
    /// do: random dest_id/src_id/seq, random bit-flip corruption positions
    /// and counts.
    ///
    /// This is a real (not theoretical) divergence between the two
    /// formats, measured with a 5000-trial version of this same sweep
    /// before writing the assertions below: `ProtectedHeader` never once
    /// reported `ok` with a wrong dest_id/src_id/seq/text (0/5000).
    /// `Legacy` did, on ~7% of trials (361/5000) -- because its header
    /// fields carry zero redundancy and the CRC only covers the payload,
    /// a corruption confined to header bytes (quite possible by chance
    /// with a handful of scattered bit flips) is invisible to every check
    /// this format runs. The *text* is still always correct-or-cleanly-
    /// failed either way (CRC does cover that) -- so the two formats are
    /// tested to two different, honest standards here rather than the
    /// stronger one being asserted for both and silently skipped for
    /// `Legacy` in practice.
    #[test]
    fn random_corruption_sweep_never_reports_wrong_fields() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let text = "fuzz test message with enough length to span several bytes";
        for format in [FrameFormat::Legacy, FrameFormat::ProtectedHeader] {
            for _ in 0..300 {
                let dest_id: u8 = rng.random_range(0..=254); // avoid BROADCAST_ID so addressing stays meaningful
                let src_id: u8 = rng.random();
                let seq: u8 = rng.random();
                let frame = build_frame(
                    text,
                    &BuildOptions {
                        dest_id,
                        src_id,
                        seq,
                        frame_format: format,
                        ..Default::default()
                    },
                )
                .unwrap();

                let mut corrupted = frame.clone();
                let n_flips = rng.random_range(0..=6);
                for _ in 0..n_flips {
                    let idx = rng.random_range(0..corrupted.len());
                    let bit: u8 = 1 << rng.random_range(0..8);
                    corrupted[idx] ^= bit;
                }

                let result = parse_frame(&corrupted, fec::DEFAULT_PARITY_BYTES, true, None, None);
                if result.ok {
                    // Holds for both formats: text is always FEC+CRC-verified.
                    assert_eq!(
                        result.text.as_deref(),
                        Some(text),
                        "format={format:?} dest_id={dest_id} src_id={src_id} seq={seq} n_flips={n_flips}"
                    );
                    // Metadata correctness is only guaranteed for
                    // ProtectedHeader (RS-protected header) -- see this
                    // test's doc comment for the measured Legacy exception.
                    if format == FrameFormat::ProtectedHeader {
                        assert_eq!(
                            result.dest_id,
                            Some(dest_id),
                            "format={format:?} n_flips={n_flips}"
                        );
                        assert_eq!(
                            result.src_id,
                            Some(src_id),
                            "format={format:?} n_flips={n_flips}"
                        );
                        assert_eq!(result.seq, Some(seq), "format={format:?} n_flips={n_flips}");
                    }
                }
                // else: clean failure, acceptable -- never allowed to be "ok" with wrong text
            }
        }
    }

    #[test]
    fn missing_parity_start_marker_reported_not_crashed() {
        let frame = build_with_format("hi", FrameFormat::Legacy);
        let truncated = &frame[..5];
        let result = parse(truncated);
        assert!(!result.ok);
        assert!(result.text.is_none());
    }

    // --- ProtectedHeader vs Legacy: the actual improvement --------------

    /// The concrete case that motivated `ProtectedHeader`: a corrupted
    /// marker byte (simulating what real AMR-NB at 4.75kbps produced)
    /// takes down a `Legacy` frame outright ("PARITY_START marker not
    /// found") even though the payload's own FEC budget is untouched, but
    /// the same corruption *pattern* applied to a `ProtectedHeader` frame's
    /// analogous region either gets corrected by the header RS or, if it
    /// happens to land in the payload instead, is still within that
    /// frame's normal FEC budget -- because there's no separate marker
    /// byte there left to corrupt.
    #[test]
    fn protected_header_survives_marker_position_corruption_that_breaks_legacy() {
        let text = "hi";
        let legacy = build_with_format(text, FrameFormat::Legacy);
        // Corrupt exactly the byte at FEC_PARITY_START's position.
        let parity_start_idx = legacy
            .iter()
            .position(|&b| b == codes::FEC_PARITY_START)
            .unwrap();
        let mut corrupted_legacy = legacy.clone();
        corrupted_legacy[parity_start_idx] = 0x01;
        let legacy_result = parse_frame(
            &corrupted_legacy,
            fec::DEFAULT_PARITY_BYTES,
            true,
            None,
            None,
        );
        assert!(!legacy_result.ok);
        assert!(
            legacy_result.reason.contains("PARITY_START"),
            "{legacy_result:?}"
        );

        // ProtectedHeader has no such marker to corrupt in the first
        // place -- confirm a clean round trip still works for the same text.
        let protected = build_with_format(text, FrameFormat::ProtectedHeader);
        let protected_result = parse_frame(&protected, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(protected_result.ok);
        assert_eq!(protected_result.text.as_deref(), Some(text));
    }

    #[test]
    fn protected_header_corrects_corrupted_header_bytes() {
        let text = "hello protected header";
        let mut frame = build_with_format(text, FrameFormat::ProtectedHeader);
        // Corrupt 2 bytes within the header+header-parity region (right
        // after SOF+SCHEME_ID_LO) -- well within HEADER_PARITY_BYTES=12's
        // t=6 correction budget.
        frame[2] ^= 0xFF;
        frame[4] ^= 0xFF;
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(result.ok, "{result:?}");
        assert_eq!(result.text.as_deref(), Some(text));
    }

    #[test]
    fn protected_header_correction_recovers_src_id_too() {
        let text = "hello protected header";
        let mut frame = build_frame(
            text,
            &BuildOptions {
                src_id: 99,
                frame_format: FrameFormat::ProtectedHeader,
                ..Default::default()
            },
        )
        .unwrap();
        // Same corruption pattern as the sibling test above -- within the
        // t=6 budget -- but this time checking that SRC_ID specifically
        // (not just the overall `ok`/text) comes back correct, since a
        // header-wide RS codeword recovering "some" fields right while
        // silently getting SRC_ID wrong would be exactly the kind of
        // partial-corruption bug this codebase's CRC/FEC layers exist to
        // rule out elsewhere.
        frame[2] ^= 0xFF;
        frame[4] ^= 0xFF;
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(result.ok, "{result:?}");
        assert_eq!(result.src_id, Some(99));
        assert_eq!(result.text.as_deref(), Some(text));
    }

    #[test]
    fn protected_header_fails_cleanly_beyond_its_correction_budget() {
        let text = "hello protected header";
        let mut frame = build_with_format(text, FrameFormat::ProtectedHeader);
        // Corrupt more header bytes than HEADER_PARITY_BYTES=12 (t=6) can fix.
        for byte in &mut frame[2..16] {
            *byte ^= 0xFF;
        }
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(!result.ok || result.text.as_deref() == Some(text));
    }

    /// Found live against a real acoustic channel: a live poll scanning a
    /// still-growing capture buffer can catch a transmission with its
    /// header genuinely cut short (the rest hasn't arrived yet) -- distinct
    /// from a header that arrived complete but failed RS correction. Before
    /// this test existed both cases reported the identical
    /// "protected header FEC uncorrectable" string, which a live poll loop
    /// can't safely treat as "wait, more is coming" (it's also the string a
    /// genuinely corrupt header produces) -- so a header truncated by buffer
    /// boundaries got reported as a hard failure and permanently skipped,
    /// even though the same audio, captured in full, decodes cleanly (see
    /// this repository's README for how this was found).
    #[test]
    fn truncated_protected_header_is_reported_distinctly_from_uncorrectable() {
        let text = "hello protected header";
        let frame = build_with_format(text, FrameFormat::ProtectedHeader);
        // Cut the wire short while still inside the header+header-parity
        // region (right after SOF+SCHEME_ID_LO) -- simulates a live buffer
        // that hasn't captured the rest of the transmission yet.
        let truncated = &frame[..frame.len().min(6)];
        let result = parse_frame(truncated, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(!result.ok);
        assert_eq!(result.reason, "unexpected end of frame reading protected header");

        // A genuinely corrupt but *complete* header still reports the
        // original, distinct "uncorrectable" reason -- this test would be
        // meaningless if both cases collapsed to the same string again.
        let mut corrupted = frame.clone();
        for byte in &mut corrupted[2..16] {
            *byte ^= 0xFF;
        }
        let corrupted_result = parse_frame(&corrupted, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(!corrupted_result.ok);
        assert_ne!(
            corrupted_result.reason,
            "unexpected end of frame reading protected header"
        );
    }

    #[test]
    fn frame_wire_length_works_for_both_formats() {
        for format in [FrameFormat::Legacy, FrameFormat::ProtectedHeader] {
            let frame = build_with_format("hello", format);
            let len = frame_wire_length(&frame, 0, fec::DEFAULT_PARITY_BYTES);
            assert_eq!(len, Some(frame.len()), "format={format:?}");
        }
    }

    #[test]
    fn frame_wire_length_correctly_finds_second_frame_for_both_formats() {
        for format in [FrameFormat::Legacy, FrameFormat::ProtectedHeader] {
            let frame1 = build_with_format("first", format);
            let frame2 = build_with_format("second", format);
            let mut both = frame1.clone();
            both.extend(&frame2);
            let len1 = frame_wire_length(&both, 0, fec::DEFAULT_PARITY_BYTES).unwrap();
            assert_eq!(len1, frame1.len(), "format={format:?}");
            let result2 = parse_frame(&both[len1..], fec::DEFAULT_PARITY_BYTES, true, None, None);
            assert!(result2.ok, "format={format:?} {result2:?}");
            assert_eq!(result2.text.as_deref(), Some("second"));
        }
    }

    /// dest_id/src_id are `u8`s -- nothing stops a value that happens to
    /// equal a *reserved code* (e.g. `FEC_SOF`, `ESCAPE` itself), which
    /// every prior addressing/src_id test avoided by accident (all used
    /// small values like 5/7/42/99). `stuff()` is proven correct at the
    /// byte level (`framing::tests::round_trip_stuff_unstuff` covers all
    /// 256 values), but this checks the actual frame-level consequence:
    /// escaping a header byte changes the frame's wire length, which
    /// `frame_wire_length`'s multi-frame scanning depends on getting
    /// exactly right.
    #[test]
    fn reserved_value_dest_and_src_id_still_round_trip_and_scan_correctly() {
        for format in [FrameFormat::Legacy, FrameFormat::ProtectedHeader] {
            let frame1 = build_frame(
                "first",
                &BuildOptions {
                    dest_id: codes::FEC_SOF, // 144: a reserved code value, needs escaping
                    src_id: codes::ESCAPE,   // 147: ESCAPE itself
                    frame_format: format,
                    ..Default::default()
                },
            )
            .unwrap();
            let frame2 = build_with_format("second", format);
            let mut both = frame1.clone();
            both.extend(&frame2);

            let len1 = frame_wire_length(&both, 0, fec::DEFAULT_PARITY_BYTES)
                .unwrap_or_else(|| panic!("format={format:?}: frame_wire_length failed"));
            assert_eq!(len1, frame1.len(), "format={format:?}");

            let result1 = parse_frame(&both[..len1], fec::DEFAULT_PARITY_BYTES, true, None, None);
            assert!(result1.ok, "format={format:?} {result1:?}");
            assert_eq!(result1.text.as_deref(), Some("first"), "format={format:?}");
            assert_eq!(result1.dest_id, Some(codes::FEC_SOF), "format={format:?}");
            assert_eq!(result1.src_id, Some(codes::ESCAPE), "format={format:?}");

            let result2 = parse_frame(&both[len1..], fec::DEFAULT_PARITY_BYTES, true, None, None);
            assert!(result2.ok, "format={format:?} {result2:?}");
            assert_eq!(result2.text.as_deref(), Some("second"), "format={format:?}");
        }
    }

    // --- Addressing (dest_id) -----------------------------------------

    #[test]
    fn default_dest_id_is_broadcast() {
        let frame = build("hi");
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, Some(42), None);
        assert!(result.ok);
        assert_eq!(result.dest_id, Some(BROADCAST_ID));
    }

    #[test]
    fn addressed_frame_accepted_by_intended_recipient() {
        let frame = build_frame(
            "for bob",
            &BuildOptions {
                dest_id: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, Some(7), None);
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some("for bob"));
        assert_eq!(result.dest_id, Some(7));
    }

    #[test]
    fn addressed_frame_rejected_by_other_recipient_without_running_fec() {
        let frame = build_frame(
            "for bob",
            &BuildOptions {
                dest_id: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, Some(99), None);
        assert!(!result.ok);
        assert_eq!(result.dest_id, Some(7));
        assert!(result.reason.contains("not addressed to me"));
    }

    #[test]
    fn no_my_id_means_everyone_accepts_regardless_of_dest_id() {
        let frame = build_frame(
            "for bob",
            &BuildOptions {
                dest_id: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse(&frame); // my_id not given -> no filtering
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some("for bob"));
    }

    // --- Sender identification (src_id) -----------------------------------

    #[test]
    fn default_src_id_is_unknown() {
        let frame = build("hi");
        let result = parse(&frame);
        assert!(result.ok);
        assert_eq!(result.src_id, Some(UNKNOWN_SRC_ID));
    }

    #[test]
    fn src_id_roundtrips_independent_of_dest_id_for_both_formats() {
        for format in [FrameFormat::Legacy, FrameFormat::ProtectedHeader] {
            let frame = build_frame(
                "group chat message",
                &BuildOptions {
                    dest_id: BROADCAST_ID,
                    src_id: 42,
                    frame_format: format,
                    ..Default::default()
                },
            )
            .unwrap();
            let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, None, None);
            assert!(result.ok, "format={format:?} {result:?}");
            assert_eq!(result.dest_id, Some(BROADCAST_ID), "format={format:?}");
            assert_eq!(result.src_id, Some(42), "format={format:?}");
        }
    }

    #[test]
    fn not_for_me_result_still_reports_src_id() {
        let frame = build_frame(
            "for bob",
            &BuildOptions {
                dest_id: 7,
                src_id: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, Some(99), None);
        assert!(!result.ok);
        assert_eq!(result.src_id, Some(3));
    }

    // --- Encryption ------------------------------------------------------

    fn session_keypair() -> ([u8; 32], [u8; 32]) {
        let (alice_priv, alice_pub) = crypto::generate_keypair();
        let (bob_priv, bob_pub) = crypto::generate_keypair();
        let alice_key = crypto::derive_session_key(&alice_priv, &bob_pub);
        let bob_key = crypto::derive_session_key(&bob_priv, &alice_pub);
        assert_eq!(alice_key, bob_key);
        (alice_key, bob_key)
    }

    #[test]
    fn encrypted_roundtrip() {
        let (alice_key, bob_key) = session_keypair();
        let text = "Secret message 秘密 \u{1F600}";
        let frame = build_frame(
            text,
            &BuildOptions {
                session_key: Some(&alice_key),
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse_frame(
            &frame,
            fec::DEFAULT_PARITY_BYTES,
            true,
            None,
            Some(&bob_key),
        );
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some(text));
    }

    /// SRC_ID lives in the plaintext header, not the encrypted payload --
    /// this locks in that it's still readable (for group-chat sender
    /// attribution) on an otherwise-encrypted frame, for both formats.
    #[test]
    fn encrypted_frame_carries_src_id_for_both_formats() {
        for format in [FrameFormat::Legacy, FrameFormat::ProtectedHeader] {
            let (alice_key, bob_key) = session_keypair();
            let frame = build_frame(
                "secret group message",
                &BuildOptions {
                    src_id: 5,
                    session_key: Some(&alice_key),
                    frame_format: format,
                    ..Default::default()
                },
            )
            .unwrap();
            let result = parse_frame(
                &frame,
                fec::DEFAULT_PARITY_BYTES,
                true,
                None,
                Some(&bob_key),
            );
            assert!(result.ok, "format={format:?} {result:?}");
            assert_eq!(result.src_id, Some(5), "format={format:?}");
        }
    }

    #[test]
    fn encrypted_frame_without_session_key_fails_cleanly() {
        let (alice_key, _bob_key) = session_keypair();
        let frame = build_frame(
            "secret",
            &BuildOptions {
                session_key: Some(&alice_key),
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse(&frame); // no session_key given
        assert!(!result.ok);
    }

    #[test]
    fn encrypted_frame_with_wrong_key_fails_cleanly() {
        let (alice_key, _bob_key) = session_keypair();
        let (unrelated_key, _) = session_keypair(); // a session key from a different key exchange entirely
        let frame = build_frame(
            "secret",
            &BuildOptions {
                session_key: Some(&alice_key),
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse_frame(
            &frame,
            fec::DEFAULT_PARITY_BYTES,
            true,
            None,
            Some(&unrelated_key),
        );
        assert!(!result.ok);
        assert!(result.text.is_none());
    }

    #[test]
    fn encrypted_payload_is_not_readable_plaintext_on_the_wire() {
        let (alice_key, _bob_key) = session_keypair();
        let text = "PLAINTEXT_MARKER_XYZ";
        let frame = build_frame(
            text,
            &BuildOptions {
                session_key: Some(&alice_key),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!frame.windows(text.len()).any(|w| w == text.as_bytes()));
    }

    #[test]
    fn encryption_and_addressing_combine() {
        let (alice_key, bob_key) = session_keypair();
        let frame = build_frame(
            "for bob, secretly",
            &BuildOptions {
                dest_id: 7,
                session_key: Some(&alice_key),
                ..Default::default()
            },
        )
        .unwrap();

        let eve_result = parse_frame(
            &frame,
            fec::DEFAULT_PARITY_BYTES,
            true,
            Some(99),
            Some(&bob_key),
        );
        assert!(!eve_result.ok); // wrong recipient, rejected before decryption is even attempted

        let bob_result = parse_frame(
            &frame,
            fec::DEFAULT_PARITY_BYTES,
            true,
            Some(7),
            Some(&bob_key),
        );
        assert!(bob_result.ok);
        assert_eq!(bob_result.text.as_deref(), Some("for bob, secretly"));
    }

    // --- Sequencing (seq / more_frames) -----------------------------------

    #[test]
    fn default_seq_and_more_frames() {
        let frame = build("hi");
        let result = parse(&frame);
        assert!(result.ok);
        assert_eq!(result.seq, Some(0));
        assert_eq!(result.more_frames, Some(false));
    }

    #[test]
    fn seq_and_more_frames_roundtrip() {
        let frame = build_frame(
            "part one",
            &BuildOptions {
                seq: 3,
                more_frames: true,
                ..Default::default()
            },
        )
        .unwrap();
        let result = parse(&frame);
        assert!(result.ok);
        assert_eq!(result.seq, Some(3));
        assert_eq!(result.more_frames, Some(true));
    }

    #[test]
    fn seq_survives_corruption_alongside_other_header_fields() {
        let text = "The quick brown fox jumps over the lazy dog";
        let mut corrupted = build_frame(
            text,
            &BuildOptions {
                parity_bytes: 10,
                seq: 5,
                more_frames: true,
                ..Default::default()
            },
        )
        .unwrap();
        for offset in [6, 12, 18] {
            let i = corrupted.len().saturating_sub(offset);
            corrupted[i] ^= 0x01;
        }
        let result = parse_frame(&corrupted, 10, true, None, None);
        assert!(result.ok, "{result:?}");
        assert_eq!(result.seq, Some(5));
        assert_eq!(result.more_frames, Some(true));
    }

    // --- NackFrame (resend requests) --------------------------------------

    #[test]
    fn nack_roundtrip() {
        let nack = NackFrame {
            dest_id: 7,
            src_id: 3,
            target_id: [0xAB, 0xCD, 0xEF],
        };
        let frame = build_nack_frame(&nack);
        let parsed = parse_nack_frame(&frame).unwrap();
        assert_eq!(parsed, nack);
    }

    /// A NACK frame must never be misread as an ordinary data frame (or
    /// vice versa) -- they share the same SOF but diverge on the very next
    /// byte ([`codes::FEC_SCHEME_ID_HI`] vs [`codes::FEC_SCHEME_ID_LO`]/
    /// anything else), so dispatch must route each to its own parser only.
    #[test]
    fn nack_frame_is_not_parsed_as_a_data_frame_and_vice_versa() {
        let nack = build_nack_frame(&NackFrame {
            dest_id: 1,
            src_id: 2,
            target_id: [1, 2, 3],
        });
        let data_result = parse_frame(&nack, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(!data_result.ok);

        let data_frame = build("hello");
        assert!(parse_nack_frame(&data_frame).is_none());
    }

    #[test]
    fn nack_survives_corruption_within_header_fec_budget() {
        let nack = NackFrame {
            dest_id: 42,
            src_id: 99,
            target_id: [0x11, 0x22, 0x33],
        };
        let mut frame = build_nack_frame(&nack);
        // 2 corrupted bytes, well within NACK_PARITY_BYTES=12's t=6 budget
        // -- same margin `protected_header_corrects_corrupted_header_bytes`
        // exercises for the ordinary header.
        frame[2] ^= 0xFF;
        frame[4] ^= 0xFF;
        let parsed = parse_nack_frame(&frame).unwrap();
        assert_eq!(parsed, nack);
    }

    #[test]
    fn nack_fails_cleanly_beyond_its_correction_budget() {
        let nack = NackFrame {
            dest_id: 42,
            src_id: 99,
            target_id: [0x11, 0x22, 0x33],
        };
        let mut frame = build_nack_frame(&nack);
        for byte in &mut frame[2..15] {
            *byte ^= 0xFF;
        }
        let parsed = parse_nack_frame(&frame);
        // Must not silently hand back a wrong target_id as if it were
        // correct -- either it fails, or it's exactly right.
        assert!(parsed.is_none() || parsed == Some(nack));
    }

    #[test]
    fn nack_frame_wire_length_lets_scanning_find_the_next_frame() {
        let nack = build_nack_frame(&NackFrame {
            dest_id: 1,
            src_id: 2,
            target_id: [9, 9, 9],
        });
        let data = build("second frame");
        let mut both = nack.clone();
        both.extend(&data);

        let len1 = frame_wire_length(&both, 0, fec::DEFAULT_PARITY_BYTES).unwrap();
        assert_eq!(len1, nack.len());

        let result2 = parse_frame(&both[len1..], fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert!(result2.ok, "{result2:?}");
        assert_eq!(result2.text.as_deref(), Some("second frame"));
    }

    #[test]
    fn nack_reserved_value_ids_still_round_trip() {
        // Same motivation as reserved_value_dest_and_src_id_still_round_trip_and_scan_correctly:
        // dest_id/src_id/target_id are plain u8s, nothing stops them from
        // colliding with a reserved code value.
        let nack = NackFrame {
            dest_id: codes::FEC_SOF,
            src_id: codes::ESCAPE,
            target_id: [codes::FEC_SCHEME_ID_HI, codes::NACK, codes::ACK],
        };
        let frame = build_nack_frame(&nack);
        let parsed = parse_nack_frame(&frame).unwrap();
        assert_eq!(parsed, nack);
    }

    // --- preview_frame_protected (live, pre-FEC decode readout) -------------

    #[test]
    fn preview_matches_final_decode_once_the_whole_frame_has_arrived() {
        let frame = build("hello world");
        let preview = preview_frame_protected(&frame, true, None).unwrap();
        assert_eq!(preview.text, "hello world");
        assert_eq!(preview.bytes_seen, preview.declared_len);

        let result = parse_frame(&frame, fec::DEFAULT_PARITY_BYTES, true, None, None);
        assert_eq!(preview.text, result.text.unwrap());
    }

    #[test]
    fn preview_grows_as_more_of_the_payload_arrives() {
        let frame = build("hello world, this is a longer message");
        let full = preview_frame_protected(&frame, true, None).unwrap();

        // Truncate to roughly half the frame -- on a clean (uncorrupted)
        // channel the raw bytes seen so far must be an honest prefix of
        // what the full frame eventually decodes to.
        let half = &frame[..frame.len() / 2];
        let partial = preview_frame_protected(half, true, None)
            .expect("header is well within the first half of a frame this long");
        assert!(partial.bytes_seen < partial.declared_len);
        assert!(full.text.starts_with(&partial.text));
    }

    #[test]
    fn preview_is_none_before_the_header_has_arrived() {
        let frame = build("hello");
        // Cut it off partway through the still-being-read protected header
        // block -- not enough tokens yet for `read_protected_header` to
        // even attempt RS correction.
        let truncated = &frame[..3];
        assert!(preview_frame_protected(truncated, true, None).is_none());
    }

    #[test]
    fn preview_is_none_for_legacy_frames() {
        let frame = build_with_format("hello", FrameFormat::Legacy);
        assert!(preview_frame_protected(&frame, true, None).is_none());
    }

    #[test]
    fn preview_is_none_when_not_addressed_to_me() {
        let frame = build_frame(
            "hello",
            &BuildOptions {
                dest_id: 5,
                ..BuildOptions::default()
            },
        )
        .unwrap();
        assert!(preview_frame_protected(&frame, true, Some(9)).is_none());
        assert!(preview_frame_protected(&frame, true, Some(5)).is_some());
        assert!(preview_frame_protected(&frame, true, None).is_some());
    }

    #[test]
    fn preview_is_none_for_encrypted_frames() {
        let key = [7u8; 32];
        let frame = build_frame(
            "secret",
            &BuildOptions {
                session_key: Some(&key),
                ..BuildOptions::default()
            },
        )
        .unwrap();
        assert!(preview_frame_protected(&frame, true, None).is_none());
    }

    #[test]
    fn preview_never_panics_on_an_arbitrarily_truncated_frame() {
        let frame = build("a reasonably long message to truncate at every possible point");
        for cut in 0..=frame.len() {
            let _ = preview_frame_protected(&frame[..cut], true, None);
        }
    }
}
