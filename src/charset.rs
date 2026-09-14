//! ASCII/UTF-8 text <-> code-stream conversion.
//!
//! Plain ASCII (0-127) is sent as a bare code -- it never collides with the
//! reserved range (129+), so it needs no escaping. Any character outside
//! ASCII is sent as its UTF-8 byte sequence wrapped in START/CONT/END flags,
//! with each raw UTF-8 byte individually stuffed (see [`crate::framing`])
//! since those bytes land in 128-255 and routinely collide with reserved
//! codes.
//!
//! Wire shape per non-ASCII character (n UTF-8 bytes):
//! `129(START) <byte1> 141(CONT) <byte2> [141 <byte3>] [141 <byte4>] 143(END)`
//!
//! The flags are kept even though UTF-8's own lead byte is self-describing,
//! because explicit boundaries let the receiver resync after a single lost
//! or corrupted code instead of relying on implicit length inference, which
//! is fragile on a lossy channel.

use crate::codes;
use crate::framing::{stuff, ReadError, Token, TokenReader};

pub fn encode_text(text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in text.chars() {
        let mut buf = [0u8; 4];
        let raw = ch.encode_utf8(&mut buf).as_bytes();
        if raw.len() == 1 && raw[0] < 128 {
            out.push(raw[0]);
            continue;
        }
        let stuffed_bytes: Vec<Vec<u8>> = raw.iter().map(|&b| stuff(&[b])).collect();
        out.push(codes::UTF8_START);
        out.extend(&stuffed_bytes[0]);
        for sb in &stuffed_bytes[1..] {
            out.push(codes::UTF8_CONT);
            out.extend(sb);
        }
        out.push(codes::UTF8_END);
    }
    out
}

/// Byte count implied by a UTF-8 lead byte's own bit pattern, or 0 if it
/// isn't a valid multi-byte lead. Used as a resync fallback: if the
/// UTF8_END flag itself is the thing that gets lost/corrupted, the START
/// flag alone still can't be trusted to end the sequence -- but the lead
/// byte tells us exactly how many bytes to expect regardless.
fn utf8_expected_len(lead_byte: u8) -> usize {
    if lead_byte & 0b1110_0000 == 0b1100_0000 {
        2
    } else if lead_byte & 0b1111_0000 == 0b1110_0000 {
        3
    } else if lead_byte & 0b1111_1000 == 0b1111_0000 {
        4
    } else {
        0
    }
}

/// Incremental IDLE / UTF8_ACTIVE state machine: feed it one token at a time
/// (as produced by [`TokenReader::read`]) and it returns a completed
/// character when one finishes, or `None` otherwise.
///
/// A malformed sequence (unexpected flag, escape at end of stream) aborts
/// only the character in progress and resumes at the next clean boundary,
/// so a single corrupted code doesn't take down the rest of the message.
///
/// Completion is driven primarily by reaching the lead byte's own expected
/// length ([`utf8_expected_len`]), not by waiting for UTF8_END -- relying on
/// END alone means a single lost END flag causes every following byte to be
/// swallowed into the buffer forever, since nothing else ever signals
/// "character over." Using the self-describing length as the primary
/// trigger and END as a redundant confirmation recovers cleanly even when
/// END itself is the corrupted part.
#[derive(Default)]
pub struct Utf8CharDecoder {
    buf: Vec<u8>,
    expected_len: usize,
    in_char: bool,
}

impl Utf8CharDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn flush(&mut self) -> Option<char> {
        let text = std::str::from_utf8(&self.buf)
            .ok()
            .and_then(|s| s.chars().next());
        self.buf.clear();
        text
    }

    pub fn feed(&mut self, token: Token) -> Option<char> {
        if !self.in_char {
            return match token {
                Token::Data(v) => Some(char::from(v)),
                Token::Flag(v) if v == codes::UTF8_START => {
                    self.in_char = true;
                    self.buf.clear();
                    self.expected_len = 0;
                    None
                }
                // stray CONT/END/other flag while idle: nothing to recover, skip
                Token::Flag(_) => None,
            };
        }

        // in_char == true
        match token {
            Token::Data(v) => {
                if self.buf.is_empty() {
                    self.expected_len = utf8_expected_len(v);
                    if self.expected_len == 0 {
                        self.in_char = false; // corrupted lead byte, abandon char
                        return None;
                    }
                }
                self.buf.push(v);
                if self.buf.len() >= self.expected_len {
                    self.in_char = false;
                    return self.flush();
                }
                None
            }
            Token::Flag(v) if v == codes::UTF8_CONT => None, // separator; next token is the data byte
            Token::Flag(v) if v == codes::UTF8_END => {
                self.in_char = false;
                if self.buf.is_empty() {
                    None
                } else {
                    self.flush()
                }
            }
            Token::Flag(v) if v == codes::UTF8_START => {
                // previous character was truncated by an error; restart clean
                self.buf.clear();
                self.expected_len = 0;
                None
            }
            Token::Flag(_) => {
                // foreign flag leaking into char data: abort this char
                self.in_char = false;
                self.buf.clear();
                None
            }
        }
    }
}

pub fn decode_codes(code_stream: &[u8]) -> String {
    let mut reader = TokenReader::new(code_stream);
    let mut decoder = Utf8CharDecoder::new();
    let mut result = String::new();
    while reader.has_more() {
        let token = match reader.read() {
            Ok(t) => t,
            Err(ReadError::EscapeAtEndOfStream) => break,
            Err(ReadError::Eof) => break,
        };
        if let Some(ch) = decoder.feed(token) {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trips_bare() {
        let encoded = encode_text("Hello, World! 123");
        assert!(encoded.iter().all(|&b| b < 128));
        assert_eq!(decode_codes(&encoded), "Hello, World! 123");
    }

    #[test]
    fn two_byte_utf8_round_trips() {
        let text = "héllo";
        assert_eq!(decode_codes(&encode_text(text)), text);
    }

    #[test]
    fn three_byte_utf8_round_trips() {
        let text = "你好";
        assert_eq!(decode_codes(&encode_text(text)), text);
    }

    #[test]
    fn four_byte_utf8_round_trips() {
        let text = "😀🚀";
        assert_eq!(decode_codes(&encode_text(text)), text);
    }

    #[test]
    fn mixed_ascii_and_multibyte_round_trips() {
        let text = "Hello 你好 world 😀!";
        assert_eq!(decode_codes(&encode_text(text)), text);
    }

    #[test]
    fn extreme_mixed_scripts_round_trip() {
        let text = "Ω русский العربية 👨‍👩‍👧‍👦 🇺🇸 é\u{0301}";
        assert_eq!(decode_codes(&encode_text(text)), text);
    }

    #[test]
    fn ascii_bytes_are_not_escaped() {
        assert_eq!(encode_text("A"), vec![b'A']);
    }

    /// If UTF8_END itself is dropped, completion still happens via the lead
    /// byte's self-describing length (see `utf8_expected_len`) -- so the
    /// message recovers exactly, not just partially.
    #[test]
    fn lost_end_flag_still_recovers_fully() {
        let code_stream = encode_text("a文b");
        let end_idx = code_stream
            .iter()
            .position(|&b| b == codes::UTF8_END)
            .unwrap();
        let mut corrupted = code_stream[..end_idx].to_vec();
        corrupted.extend(&code_stream[end_idx + 1..]);
        assert_eq!(decode_codes(&corrupted), "a文b");
    }

    /// If a data byte (not a flag) is lost, the character can't be
    /// reconstructed -- but the stream still resyncs at the next boundary
    /// instead of corrupting everything after it.
    #[test]
    fn lost_continuation_byte_drops_only_that_char() {
        let code_stream = encode_text("a文b");
        let start_idx = code_stream
            .iter()
            .position(|&b| b == codes::UTF8_START)
            .unwrap();
        // Drop the byte immediately after START (the first UTF-8 byte of 文).
        let mut corrupted = code_stream[..start_idx + 1].to_vec();
        corrupted.extend(&code_stream[start_idx + 2..]);
        let result = decode_codes(&corrupted);
        assert!(result.starts_with('a'));
        assert!(result.ends_with('b'));
        assert!(!result.contains('文'));
    }

    #[test]
    fn empty_input_decodes_to_empty_string() {
        assert_eq!(decode_codes(&[]), "");
    }
}
