//! Reserved control-code constants shared by every protocol layer.
//!
//! 129/141/143 mark UTF-8 multi-byte character boundaries ([`crate::charset`]).
//! 144-157 is a block reserved for frame control / FEC / CRC / ACK-NACK
//! ([`crate::protocol`], [`crate::fec`]). 158-159 mark dictionary word
//! substitutions ([`crate::dictionary`]). ESCAPE (147) drives byte-stuffing
//! ([`crate::framing`]) so that literal data bytes landing in this range
//! can't be confused with the control codes themselves.
//!
//! Every code is a plain byte value (0-255): the wire alphabet is "any byte,
//! plus these reserved markers", so `u8` is the natural type throughout.

// UTF-8 multi-byte boundary flags
pub const UTF8_START: u8 = 129;
pub const UTF8_CONT: u8 = 141;
pub const UTF8_END: u8 = 143;

// 144-157: frame control / FEC / CRC / ACK-NACK reserved block
pub const FEC_SOF: u8 = 144;
/// Bare flag immediately after [`FEC_SOF`] identifies a
/// [`crate::protocol::FrameFormat::ProtectedHeader`] frame (RS-protected
/// header, no PARITY_START/PARITY_END scan needed) -- a plain
/// [`crate::protocol::FrameFormat::Legacy`] frame has a stuffed DEST_ID
/// byte there instead, which -- because `stuff()` never emits a reserved
/// code bare -- can never be confused with this marker even under channel
/// corruption of the DEST_ID byte's *value* (only a corruption of the
/// wire's escape *structure* itself could cause ambiguity here, the same
/// residual risk every stuffed region already carries).
pub const FEC_SCHEME_ID_LO: u8 = 145;
pub const FEC_SCHEME_ID_HI: u8 = 146; // reserved for a future third frame format
pub const ESCAPE: u8 = 147;
pub const CRC_MARKER_HI: u8 = 148;
pub const CRC_MARKER_LO: u8 = 149;
pub const FEC_PARITY_START: u8 = 150;
pub const FEC_PARITY_END: u8 = 152;
pub const FEC_SOFT_SUCCESS: u8 = 153;
pub const FEC_UNCORRECTABLE: u8 = 154;
pub const FEC_RESERVED: u8 = 155;
pub const ACK: u8 = 156;
pub const NACK: u8 = 157;

// 158-159: dictionary word-substitution flags (dictionary.rs). Followed by
// 3 literal ASCII letters (A-Z, never need escaping) naming the code.
pub const DICT_LOWER: u8 = 158; // word as stored in the dictionary (lowercase)
pub const DICT_TITLE: u8 = 159; // word with its first letter capitalized

pub const ESCAPE_MASK: u8 = 0x20; // XOR mask for byte-stuffing; see framing.rs

/// True for any code in the shared reserved range (UTF-8 boundary flags plus
/// the FEC_SOF..DICT_TITLE control block) that a literal data byte must never
/// be confused with.
pub fn is_reserved(code: u8) -> bool {
    matches!(code, UTF8_START | UTF8_CONT | UTF8_END) || (FEC_SOF..=DICT_TITLE).contains(&code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_is_self_consistent() {
        // ESCAPE is itself reserved, but escaping it must not collide with
        // another reserved code.
        assert!(is_reserved(ESCAPE));
    }

    #[test]
    fn escape_mask_never_collides_with_reserved_codes() {
        for code in 0u8..=255 {
            if is_reserved(code) {
                assert!(
                    !is_reserved(code ^ ESCAPE_MASK),
                    "escape mask collides for reserved code {code}"
                );
            }
        }
    }
}
