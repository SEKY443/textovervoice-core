//! Byte-stuffing for the shared reserved-code space.
//!
//! Any literal data byte (UTF-8 continuation bytes, RS parity, CRC bytes)
//! that numerically collides with a reserved control code (see
//! [`crate::codes`]) must be escaped, or the receiver cannot tell data from
//! framing. This isn't a rare edge case: UTF-8 continuation bytes are
//! always in range 128-191, which fully contains 129/141/143 and half of
//! 144-157, so real multi-byte text collides with reserved codes routinely,
//! not just as an occasional parity-byte accident.

use crate::codes;

/// Encodes raw bytes into the code stream, escaping any byte that would
/// otherwise be indistinguishable from a reserved control code.
pub fn stuff(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for &b in data {
        if codes::is_reserved(b) {
            out.push(codes::ESCAPE);
            out.push(b ^ codes::ESCAPE_MASK);
        } else {
            out.push(b);
        }
    }
    out
}

/// A token read from the wire-code stream: a literal data byte (whether it
/// arrived bare or via an escape pair), or a bare reserved flag code that
/// the caller (charset/dictionary/protocol layer) interprets according to
/// its own state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Data(u8),
    Flag(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError {
    /// An escape code appeared with nothing after it to un-escape.
    EscapeAtEndOfStream,
    /// `read()` was called with nothing left in the stream.
    Eof,
}

/// Scans a received code stream, transparently resolving escape sequences.
pub struct TokenReader<'a> {
    codes: &'a [u8],
    pos: usize,
}

impl<'a> TokenReader<'a> {
    pub fn new(codes: &'a [u8]) -> Self {
        Self { codes, pos: 0 }
    }

    pub fn has_more(&self) -> bool {
        self.pos < self.codes.len()
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub fn read(&mut self) -> Result<Token, ReadError> {
        let c = *self.codes.get(self.pos).ok_or(ReadError::Eof)?;
        self.pos += 1;
        if c == codes::ESCAPE {
            let raw = *self
                .codes
                .get(self.pos)
                .ok_or(ReadError::EscapeAtEndOfStream)?;
            self.pos += 1;
            return Ok(Token::Data(raw ^ codes::ESCAPE_MASK));
        }
        if codes::is_reserved(c) {
            return Ok(Token::Flag(c));
        }
        Ok(Token::Data(c))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnstuffError;

/// Resolves an entire stuffed wire-code region back to raw bytes -- every
/// token is treated as literal data (used for regions that are opaque bytes
/// on purpose, like RS parity or ciphertext, never text with its own flags).
pub fn unstuff_bytes(wire_codes: &[u8]) -> Result<Vec<u8>, UnstuffError> {
    let mut reader = TokenReader::new(wire_codes);
    let mut out = Vec::new();
    while reader.has_more() {
        match reader.read().map_err(|_| UnstuffError)? {
            Token::Data(v) => out.push(v),
            Token::Flag(v) => out.push(v),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stuff_passes_through_ordinary_bytes() {
        assert_eq!(stuff(b"hello"), b"hello".to_vec());
    }

    #[test]
    fn stuff_escapes_reserved_bytes() {
        let stuffed = stuff(&[codes::FEC_SOF]);
        assert_eq!(
            stuffed,
            vec![codes::ESCAPE, codes::FEC_SOF ^ codes::ESCAPE_MASK]
        );
    }

    #[test]
    fn round_trip_stuff_unstuff() {
        let data: Vec<u8> = (0u8..=255).collect();
        let stuffed = stuff(&data);
        let unstuffed = unstuff_bytes(&stuffed).unwrap();
        assert_eq!(unstuffed, data);
    }

    #[test]
    fn token_reader_reports_flags_and_data() {
        // A bare reserved code (inserted directly by a higher layer, e.g.
        // protocol.rs's frame markers -- not via stuff(), which always
        // escapes reserved bytes instead of emitting them bare).
        let wire = [65, codes::FEC_SOF, 66];
        let mut r = TokenReader::new(&wire);
        assert_eq!(r.read().unwrap(), Token::Data(65));
        assert_eq!(r.read().unwrap(), Token::Flag(codes::FEC_SOF));
        assert_eq!(r.read().unwrap(), Token::Data(66));
        assert!(!r.has_more());
    }

    #[test]
    fn escape_at_end_of_stream_errors() {
        let mut r = TokenReader::new(&[codes::ESCAPE]);
        assert_eq!(r.read(), Err(ReadError::EscapeAtEndOfStream));
    }

    #[test]
    fn read_past_end_errors() {
        let mut r = TokenReader::new(&[]);
        assert_eq!(r.read(), Err(ReadError::Eof));
    }
}
