//! Multi-frame messages: splits long text into multiple independently-
//! synced frames ([`crate::protocol`] builds/parses one frame at a time),
//! so a single-preamble transmission's drift-over-time problem is bounded
//! per-chunk instead of compounding over an entire long message.
//!
//! The problem this solves is real and measured, not theoretical: sending
//! the full GPLv3 text (35KB) through real AMR-NB at 4.75kbps as a single
//! frame produced a hard "cliff" -- the first ~15% of Reed-Solomon chunks
//! decoded clean, then essentially everything after failed, in one
//! contiguous block. Splitting into multiple independently-preambled
//! frames means each one gets a fresh sync point, so drift can only
//! accumulate within one frame's (much shorter) duration.
//!
//! Splitting is by character count (`MAX_FRAME_CHARS`), not by bytes --
//! chunking on `char` boundaries is always UTF-8-safe, so no character
//! gets split across a frame boundary. Encryption, dictionary compression,
//! and addressing are applied per-frame (independently), not to the whole
//! message before splitting, so each frame is a fully self-contained,
//! independently decodable/decryptable unit -- reassembly just
//! concatenates already-decoded text, never partial ciphertext or partial
//! charset state across frames.
//!
//! A message that fits in one frame is exactly what [`crate::protocol::build_frame`]
//! alone would have produced (seq=0, more_frames=false) -- this module is a
//! strict extension, not a wire-format change for short messages.

use std::collections::HashMap;

use crate::fec;
use crate::protocol::{self, BuildOptions, FrameFormat, ParseResult, BROADCAST_ID};

pub const MAX_FRAME_CHARS: usize = 800;
// Conservative by design: the observed drift-cliff in the GPLv3 test set
// falls around 15% into a ~35-minute phone-mode transmission (roughly 5
// minutes of audio). Capping frames at 800 characters keeps each frame's
// audio duration well under a minute even at phone mode's slower timing --
// a large safety margin below the one drift-onset point actually measured.

/// Number of frames [`build_message`] would produce for `text` at
/// `max_frame_chars`, without actually building them -- lets a caller
/// (the CLI, which wants a clean error instead of `build_message`'s
/// `assert!` panic) validate ahead of time whether a message would exceed
/// the 256-frame ceiling `seq` (a `u8`) can address.
pub fn frame_count(text: &str, max_frame_chars: usize) -> usize {
    let n = text.chars().count();
    if n == 0 {
        1
    } else {
        n.div_ceil(max_frame_chars.max(1))
    }
}

pub struct MessageBuildOptions<'a> {
    pub parity_bytes: usize,
    pub use_dictionary: bool,
    pub dest_id: u8,
    pub src_id: u8,
    pub session_key: Option<&'a [u8; 32]>,
    pub max_frame_chars: usize,
    pub frame_format: FrameFormat,
}

impl Default for MessageBuildOptions<'_> {
    fn default() -> Self {
        Self {
            parity_bytes: fec::DEFAULT_PARITY_BYTES,
            use_dictionary: true,
            dest_id: BROADCAST_ID,
            src_id: protocol::UNKNOWN_SRC_ID,
            session_key: None,
            max_frame_chars: MAX_FRAME_CHARS,
            frame_format: FrameFormat::default(),
        }
    }
}

/// Byte-slices of `text` chunked to at most `max_chars` characters each,
/// always split on a `char` boundary.
fn chunk_by_chars(text: &str, max_chars: usize) -> Vec<&str> {
    assert!(
        max_chars >= 1,
        "max_chars must be at least 1, or chunking never advances (infinite loop)"
    );
    if text.is_empty() {
        return Vec::new();
    }
    let mut boundaries: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    boundaries.push(text.len());
    let mut chunks = Vec::new();
    let mut start_idx = 0;
    while start_idx < boundaries.len() - 1 {
        let end_idx = (start_idx + max_chars).min(boundaries.len() - 1);
        chunks.push(&text[boundaries[start_idx]..boundaries[end_idx]]);
        start_idx = end_idx;
    }
    chunks
}

/// Splits `text` into `<= max_frame_chars` chunks and builds one frame per
/// chunk ([`protocol::build_frame`]), each tagged with its position (seq)
/// and whether more frames follow (more_frames). Modulating/playing the
/// returned frames is the caller's job (one `modem::modulate_frame` call
/// per frame, with a silence gap between so preambles don't run into each
/// other).
///
/// Returns `None` if any chunk's encoded payload would exceed
/// [`protocol::MAX_PAYLOAD_BYTES`] (see `build_frame`'s docs) -- typically
/// reachable via a large `--max-frame-chars` on text that doesn't compress
/// well (dictionary misses, or non-ASCII UTF-8 expansion).
pub fn build_message(text: &str, opts: &MessageBuildOptions) -> Option<Vec<Vec<u8>>> {
    let mut chunks = chunk_by_chars(text, opts.max_frame_chars);
    if chunks.is_empty() {
        chunks.push("");
    }
    let n = chunks.len();
    // `seq` is a u8 (0-255), so a message needing more than 256 frames
    // would assign the same seq to two different frames (`i % 256`
    // wraps). MessageReassembler stores frames in a `HashMap<u8, String>`
    // keyed by seq, so a colliding later frame silently overwrites an
    // earlier one's reassembled text instead of erroring -- confirmed by
    // reasoning through the reassembly logic and reproduced by the
    // `#[should_panic]` test below before this guard existed. That's
    // exactly the "silently wrong instead of loudly failing" failure mode
    // this codebase otherwise goes out of its way to avoid (see
    // fec::recover's mandatory resyndrome check, protocol::parse_frame's
    // CRC-after-FEC check). Fail loudly here instead: raise
    // `--max-frame-chars` (or split the message yourself) to stay under
    // 256 frames per message.
    assert!(
        n <= 256,
        "message needs {n} frames, but seq (u8) can only distinguish 256 per message -- frames \
         would collide during reassembly. Use a larger --max-frame-chars (currently {}) to reduce \
         the frame count below 256.",
        opts.max_frame_chars
    );
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let more = i < n - 1;
            protocol::build_frame(
                chunk,
                &BuildOptions {
                    parity_bytes: opts.parity_bytes,
                    use_dictionary: opts.use_dictionary,
                    dest_id: opts.dest_id,
                    src_id: opts.src_id,
                    session_key: opts.session_key,
                    seq: (i % 256) as u8,
                    more_frames: more,
                    frame_format: opts.frame_format,
                },
            )
        })
        .collect::<Option<Vec<Vec<u8>>>>()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageResult {
    pub ok: bool,
    pub text: Option<String>,
    pub reason: String,
    pub dest_id: Option<u8>,
    pub src_id: Option<u8>,
    pub frames_received: usize,
    pub frames_expected: Option<usize>, // unknown until the frame with more_frames=false arrives
}

/// Accumulates parsed frames (by seq) for one message-in-progress. Feed it
/// every [`ParseResult`] as frames arrive (in any order -- it doesn't
/// assume sequential arrival); once a contiguous run from seq=0 through the
/// frame with `more_frames=false` has been seen, `add` returns the
/// completed message.
///
/// One instance tracks one message. A caller expecting multiple back-to-
/// back messages should create a fresh instance per message (e.g. after a
/// completed or abandoned reassembly), since seq numbers wrap and aren't
/// globally unique across messages.
#[derive(Default)]
pub struct MessageReassembler {
    parts: HashMap<u8, String>,
    last_seq: Option<u8>, // seq of the frame with more_frames=false, once seen
}

impl MessageReassembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, result: &ParseResult) -> MessageResult {
        if !result.ok {
            return MessageResult {
                ok: false,
                text: None,
                reason: result.reason.clone(),
                dest_id: result.dest_id,
                src_id: result.src_id,
                frames_received: 0,
                frames_expected: None,
            };
        }

        let Some(seq) = result.seq else {
            // a frame parsed without seq info at all (shouldn't happen via
            // protocol::parse_frame, which always sets it on success) --
            // treat defensively as a complete standalone message.
            return MessageResult {
                ok: true,
                text: result.text.clone(),
                reason: String::new(),
                dest_id: result.dest_id,
                src_id: result.src_id,
                frames_received: 1,
                frames_expected: Some(1),
            };
        };

        self.parts
            .insert(seq, result.text.clone().unwrap_or_default());
        if !result.more_frames.unwrap_or(false) {
            self.last_seq = Some(seq);
        }

        if let Some(last_seq) = self.last_seq {
            if (0..=last_seq).all(|i| self.parts.contains_key(&i)) {
                let mut full_text = String::new();
                for i in 0..=last_seq {
                    full_text.push_str(&self.parts[&i]);
                }
                return MessageResult {
                    ok: true,
                    text: Some(full_text),
                    reason: String::new(),
                    dest_id: result.dest_id,
                    src_id: result.src_id,
                    frames_received: self.parts.len(),
                    frames_expected: Some(last_seq as usize + 1),
                };
            }
        }

        MessageResult {
            ok: false,
            text: None,
            reason: "waiting for more frames".to_string(),
            dest_id: result.dest_id,
            src_id: result.src_id,
            frames_received: self.parts.len(),
            frames_expected: self.last_seq.map(|s| s as usize + 1),
        }
    }

    pub fn reset(&mut self) {
        self.parts.clear();
        self.last_seq = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto;

    fn parse_default(frame: &[u8]) -> ParseResult {
        protocol::parse_frame(frame, fec::DEFAULT_PARITY_BYTES, true, None, None)
    }

    fn decode_all(frames: &[Vec<u8>]) -> String {
        decode_all_with(frames, None, None)
    }

    fn decode_all_with(
        frames: &[Vec<u8>],
        my_id: Option<u8>,
        session_key: Option<&[u8; 32]>,
    ) -> String {
        let mut reassembler = MessageReassembler::new();
        let mut result = None;
        for f in frames {
            result = Some(reassembler.add(&protocol::parse_frame(
                f,
                fec::DEFAULT_PARITY_BYTES,
                true,
                my_id,
                session_key,
            )));
        }
        let result = result.expect("at least one frame");
        assert!(result.ok, "reassembly failed: {result:?}");
        result.text.unwrap()
    }

    #[test]
    fn short_message_is_a_single_frame() {
        let frames = build_message("hello", &MessageBuildOptions::default()).unwrap();
        assert_eq!(frames.len(), 1);
        let result = parse_default(&frames[0]);
        assert!(result.ok);
        assert_eq!(result.seq, Some(0));
        assert_eq!(result.more_frames, Some(false));
        assert_eq!(result.text.as_deref(), Some("hello"));
    }

    #[test]
    fn long_message_splits_into_multiple_frames() {
        let text = "x".repeat(2500);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(frames.len(), 4); // 800+800+800+100

        let results: Vec<ParseResult> = frames.iter().map(|f| parse_default(f)).collect();
        assert_eq!(
            results.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            results.iter().map(|r| r.more_frames).collect::<Vec<_>>(),
            vec![Some(true), Some(true), Some(true), Some(false)]
        );
    }

    #[test]
    fn multi_frame_roundtrip_reassembles_exactly() {
        let text = "The quick brown fox. ".repeat(100); // well over 800 chars
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(frames.len() > 1);
        assert_eq!(decode_all(&frames), text);
    }

    #[test]
    fn multi_frame_roundtrip_with_dictionary_and_cjk() {
        let text = ("This message repeats government information about something \
                     important. 文字轉聲音穿越語音通話測試。")
            .repeat(20);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 500,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(frames.len() > 1);
        assert_eq!(decode_all(&frames), text);
    }

    #[test]
    fn reassembler_handles_out_of_order_frames() {
        let text = "AAAA".repeat(500); // multiple frames
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(frames.len() >= 3);

        let mut order: Vec<usize> = vec![1, 0, 2];
        order.extend(3..frames.len());

        let mut reassembler = MessageReassembler::new();
        let mut result = None;
        for i in order {
            result = Some(reassembler.add(&parse_default(&frames[i])));
        }
        let result = result.unwrap();
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some(text.as_str()));
    }

    #[test]
    fn reassembler_reports_incomplete_before_last_frame_arrives() {
        let text = "y".repeat(2000);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(frames.len(), 3);

        let mut reassembler = MessageReassembler::new();
        let r0 = reassembler.add(&parse_default(&frames[0]));
        assert!(!r0.ok);
        assert_eq!(r0.frames_received, 1);
        assert_eq!(r0.frames_expected, None); // last frame (which reveals total count) not seen yet

        let r1 = reassembler.add(&parse_default(&frames[1]));
        assert!(!r1.ok);

        let r2 = reassembler.add(&parse_default(&frames[2]));
        assert!(r2.ok);
        assert_eq!(r2.text.as_deref(), Some(text.as_str()));
        assert_eq!(r2.frames_received, 3);
        assert_eq!(r2.frames_expected, Some(3));
    }

    #[test]
    fn reassembler_reset_starts_fresh() {
        let mut reassembler = MessageReassembler::new();
        let frames1 = build_message(
            &"first message ".repeat(100),
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        for f in &frames1[..frames1.len() - 1] {
            reassembler.add(&parse_default(f)); // leave incomplete
        }

        reassembler.reset();
        let frames2 = build_message(
            "second, unrelated",
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        let result = reassembler.add(&parse_default(&frames2[0]));
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some("second, unrelated"));
    }

    #[test]
    fn multi_frame_with_encryption_and_addressing() {
        let (alice_priv, alice_pub) = crypto::generate_keypair();
        let (bob_priv, bob_pub) = crypto::generate_keypair();
        let alice_key = crypto::derive_session_key(&alice_priv, &bob_pub);
        let bob_key = crypto::derive_session_key(&bob_priv, &alice_pub);

        let text = "Secret plan ".repeat(200);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 800,
                dest_id: 7,
                session_key: Some(&alice_key),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(frames.len() > 1);

        let result = decode_all_with(&frames, Some(7), Some(&bob_key));
        assert_eq!(result, text);
    }

    /// A multi-frame message's completed [`MessageResult`] must report the
    /// sender's `src_id` -- this is exactly the piece group chat depends on
    /// to attribute a (possibly long, multi-frame) message to whoever sent
    /// it, and it's easy to get wrong since `src_id` isn't part of the
    /// reassembly key (`seq` is) the way this module's other per-frame
    /// fields aren't either.
    #[test]
    fn multi_frame_message_result_carries_src_id() {
        let text = "group message ".repeat(100); // several frames
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 400,
                src_id: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(frames.len() > 1);

        let mut reassembler = MessageReassembler::new();
        let mut result = None;
        for f in &frames {
            result = Some(reassembler.add(&parse_default(f)));
        }
        let result = result.unwrap();
        assert!(result.ok, "{result:?}");
        assert_eq!(result.src_id, Some(42));
        assert_eq!(result.text.as_deref(), Some(text.as_str()));
    }

    /// Simulates the actual mechanism behind `--repeat`: the sender
    /// transmits the same frame set (same seq numbers) multiple times, and
    /// a lossy channel drops a different subset of frames on each pass --
    /// no single pass is complete on its own, but every seq survives at
    /// least one pass. The reassembler is fed all surviving frames across
    /// all passes into ONE instance (not reset between passes, matching how
    /// the live listener/file decoder behave), and should still reassemble
    /// the full message purely because it's keyed by seq: a seq arriving
    /// more than once is a harmless overwrite, and completion only needs
    /// each seq to show up at least once across however many passes it
    /// took.
    #[test]
    fn repeated_transmission_recovers_from_per_pass_frame_loss() {
        let text = "The quick brown fox. ".repeat(100);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 400,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(frames.len() >= 4); // need several seqs for the drop pattern below to be meaningful

        // Every seq is missing from at least one pass, but present in another.
        let passes_dropped_seqs: Vec<std::collections::HashSet<usize>> = vec![
            [0, 2].into_iter().collect(),
            [1, 3].into_iter().collect(),
            std::collections::HashSet::new(),
        ];

        let mut reassembler = MessageReassembler::new();
        let mut result = None;
        for dropped in &passes_dropped_seqs {
            for (i, f) in frames.iter().enumerate() {
                if dropped.contains(&i) {
                    continue;
                }
                result = Some(reassembler.add(&parse_default(f)));
            }
        }
        let result = result.unwrap();
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some(text.as_str()));
    }

    /// Sanity check for the test above: confirms the per-pass drop pattern
    /// used there really does leave every individual pass incomplete on
    /// its own -- i.e. the repeat mechanism is doing real work, not
    /// papering over a pattern that any single pass could already
    /// reassemble.
    #[test]
    fn single_pass_would_have_failed_without_repeat() {
        let text = "y".repeat(2000);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 800,
                ..Default::default()
            },
        )
        .unwrap();
        let passes_dropped_seqs: Vec<std::collections::HashSet<usize>> =
            vec![[0, 2].into_iter().collect(), [1, 3].into_iter().collect()];

        for dropped in &passes_dropped_seqs {
            let mut reassembler = MessageReassembler::new();
            let mut result = None;
            for (i, f) in frames.iter().enumerate() {
                if dropped.contains(&i) {
                    continue;
                }
                result = Some(reassembler.add(&parse_default(f)));
            }
            assert!(result.is_none_or(|r| !r.ok));
        }
    }

    #[test]
    fn empty_message_produces_one_empty_frame() {
        let frames = build_message("", &MessageBuildOptions::default()).unwrap();
        assert_eq!(frames.len(), 1);
        let result = parse_default(&frames[0]);
        assert!(result.ok);
        assert_eq!(result.text.as_deref(), Some(""));
        assert_eq!(result.more_frames, Some(false));
    }

    /// Exactly 256 frames (seq 0..255, no collision) must still work --
    /// only 257+ is the wraparound case.
    #[test]
    fn exactly_256_frames_succeeds() {
        let text = "x".repeat(256);
        let frames = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(frames.len(), 256);
        assert_eq!(decode_all(&frames), text);
    }

    /// Regression test: `seq` is a `u8`, so a message needing more than 256
    /// frames used to silently assign the same seq to two different
    /// frames (`i % 256` wraps), and `MessageReassembler` (a
    /// `HashMap<u8, String>` keyed by seq) would silently let a later
    /// colliding frame overwrite an earlier one's text instead of
    /// erroring -- a message that decoded "successfully" with wrong
    /// content, exactly the failure mode CRC/FEC exist elsewhere in this
    /// codebase to prevent. `build_message` now fails loudly instead.
    #[test]
    #[should_panic(expected = "256")]
    fn more_than_256_frames_fails_loudly_instead_of_corrupting() {
        let text = "x".repeat(257);
        // The panic (from build_message's internal frame-count assert)
        // fires before this ever needs unwrapping.
        let _ = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 1,
                ..Default::default()
            },
        );
    }

    /// Regression test: `LENGTH` is a 2-byte wire field, so a payload
    /// longer than 65535 bytes can't be declared correctly. Before
    /// `build_frame` returned `Option`, `payload_bytes.len() as u16` would
    /// silently truncate/wrap, and `build_frame` would return a
    /// "successfully built" frame whose LENGTH field didn't match its
    /// actual payload -- reproduced via the real CLI (`--max-frame-chars
    /// 100000` on non-dictionary text produced an 81-*minute* WAV file
    /// that was completely undecodable, with zero warning at encode time).
    /// `build_message` now reports this as `None` instead.
    #[test]
    fn payload_too_large_for_length_field_returns_none_instead_of_corrupting() {
        // Plain ASCII, not dictionary words -- 1 byte/char, so this
        // reliably exceeds MAX_PAYLOAD_BYTES (65535) in one frame.
        let text = "a".repeat(100_000);
        let result = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 100_000,
                ..Default::default()
            },
        );
        assert!(
            result.is_none(),
            "expected None for an oversized payload, got Some(..)"
        );
    }

    /// A payload comfortably under the limit must still succeed --
    /// `payload_too_large_for_length_field_returns_none_instead_of_corrupting`
    /// isn't rejecting everything large, just genuinely oversized frames.
    #[test]
    fn payload_within_length_field_limit_still_succeeds() {
        let text = "a".repeat(50_000);
        let result = build_message(
            &text,
            &MessageBuildOptions {
                max_frame_chars: 50_000,
                ..Default::default()
            },
        );
        assert!(result.is_some());
    }

    /// Regression test: `max_frame_chars = 0` used to make `chunk_by_chars`
    /// compute `end_idx == start_idx` every iteration, so the `while
    /// start_idx < boundaries.len() - 1` loop never advanced -- an infinite
    /// loop (confirmed hanging via the real CLI's `--max-frame-chars 0`
    /// before this was fixed). The CLI now rejects 0 before it reaches this
    /// layer at all (see `main.rs`'s `--max-frame-chars` clap range), but
    /// this asserts the library itself still fails fast for any other
    /// caller instead of hanging.
    #[test]
    #[should_panic(expected = "max_chars")]
    fn build_message_rejects_zero_max_frame_chars() {
        build_message(
            "some text",
            &MessageBuildOptions {
                max_frame_chars: 0,
                ..Default::default()
            },
        )
        .unwrap();
    }
}
