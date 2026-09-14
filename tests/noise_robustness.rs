//! Integration tests that stress the audio-domain pipeline (modem + FEC +
//! CRC together) under synthetic channel impairment -- AWGN and burst
//! sample corruption -- rather than the wire-code-level corruption the
//! unit tests already cover in `fec::tests`/`protocol::tests`. This is the
//! kind of test the original Python prototype ran against real AMR-NB via
//! `tools/codec_validation/realism_suite.py`; this file's AWGN cases are
//! the digital-only analog of that suite's "noise robustness" experiment
//! (no ffmpeg/codec involved, so it runs anywhere `cargo test` does).
//!
//! The property under test throughout is NOT "always decodes correctly" --
//! it's "never returns wrong text as if it were correct." A frame is
//! allowed to fail (return `None`/`ok=false`) under heavy noise; it is
//! never allowed to panic or to silently hand back corrupted text, because
//! CRC-16 sits between FEC and the caller specifically to catch that.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use textovervoice_core::{message, modem, protocol};

fn awgn(audio: &mut [f64], snr_db: f64, seed: u64) {
    let signal_power: f64 = audio.iter().map(|x| x * x).sum::<f64>() / audio.len() as f64;
    let noise_power = signal_power / 10f64.powf(snr_db / 10.0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0, noise_power.sqrt()).unwrap();
    for x in audio.iter_mut() {
        *x += dist.sample(&mut rng);
    }
}

fn build_and_modulate(text: &str, parity_bytes: usize) -> Vec<f64> {
    let frames = message::build_message(
        text,
        &message::MessageBuildOptions {
            parity_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        frames.len(),
        1,
        "test texts are chosen to fit in a single frame"
    );
    let symbols = modem::bytes_to_symbols(&frames[0]);
    modem::modulate_frame(
        &symbols,
        modem::DEFAULT_SYMBOL_DURATION_S,
        modem::DEFAULT_GUARD_S,
        modem::SR,
    )
}

fn decode_audio(audio: &[f64], parity_bytes: usize) -> Option<String> {
    let reference = modem::generate_preamble(
        modem::PREAMBLE_DURATION_S,
        modem::PREAMBLE_F0,
        modem::PREAMBLE_F1,
        modem::SR,
    );
    let (offset, _score) = modem::find_preamble(audio, &reference, 0.4)?;
    let payload_start = offset + (modem::PREAMBLE_GUARD_S * modem::SR as f64) as usize;
    let step_n =
        ((modem::DEFAULT_SYMBOL_DURATION_S + modem::DEFAULT_GUARD_S) * modem::SR as f64) as usize;
    let available = audio.len().saturating_sub(payload_start);
    let n_symbols = (available / step_n).min(2000);
    let detections = modem::demodulate(
        &audio[payload_start..],
        n_symbols,
        modem::DEFAULT_SYMBOL_DURATION_S,
        modem::DEFAULT_GUARD_S,
        modem::SR,
        0,
    );
    let symbols: Vec<u8> = detections.iter().map(|d| d.symbol).collect();
    let n_bytes = (symbols.len() * modem::BITS_PER_SYMBOL as usize) / 8;
    let frame_codes = modem::symbols_to_bytes(&symbols, n_bytes);
    let result = protocol::parse_frame(&frame_codes, parity_bytes, true, None, None);
    result.ok.then_some(result.text).flatten()
}

#[test]
fn high_snr_always_exact() {
    let text = "The quick brown fox jumps over the lazy dog";
    for snr in [30.0, 20.0] {
        let mut audio = build_and_modulate(text, 10);
        awgn(&mut audio, snr, 42);
        assert_eq!(
            decode_audio(&audio, 10).as_deref(),
            Some(text),
            "failed at SNR={snr}dB (should be trivially clean)"
        );
    }
}

/// The load-bearing property: across a wide sweep of noise levels and
/// seeds, decoding never returns text other than the original -- it's
/// allowed to fail, never to lie.
#[test]
fn low_snr_never_panics_and_never_silently_corrupts() {
    let text = "Testing graceful degradation under channel noise, 1234567890";
    let mut successes = 0u32;
    let mut trials = 0u32;
    for snr in [15.0, 10.0, 5.0, 0.0, -5.0, -10.0] {
        for seed in 0..8u64 {
            let mut audio = build_and_modulate(text, 10);
            awgn(&mut audio, snr, seed);
            let result = decode_audio(&audio, 10);
            trials += 1;
            if let Some(t) = &result {
                assert_eq!(t, text, "SNR={snr}dB seed={seed} returned WRONG text (should be impossible: CRC should have caught this)");
                successes += 1;
            } // else: clean failure, acceptable
        }
    }
    // Sanity check that the test is actually exercising a real range (not
    // all-pass or all-fail, which would mean the SNR sweep wasn't doing
    // anything meaningful).
    assert!(
        successes > 0,
        "expected at least some noisy trials to still succeed"
    );
    assert!(successes < trials, "expected at least some noisy trials to fail (SNR sweep should include unrecoverable levels)");
}

/// Burst/impulse corruption (a handful of samples slammed to full scale) --
/// a different failure shape than continuous AWGN, closer to a click or a
/// dropout. Same property: never wrong, failure is fine.
#[test]
fn burst_sample_corruption_never_silently_corrupts() {
    let text = "Burst corruption test message";
    use rand::RngExt;
    for (rng_seed, n_bursts) in (7u64..).zip([1usize, 5, 20, 100]) {
        let mut audio = build_and_modulate(text, 10);
        let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
        for _ in 0..n_bursts {
            let idx = rng.random_range(0..audio.len());
            audio[idx] = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        }
        let result = decode_audio(&audio, 10);
        assert!(
            result.is_none() || result.as_deref() == Some(text),
            "n_bursts={n_bursts} returned wrong text: {result:?}"
        );
    }
}

/// The full "extreme UTF-8" style stress case from the original prototype's
/// README (CJK, RTL Arabic, Cyrillic, combining diacritics, currency/math
/// symbols, a multi-codepoint ZWJ family emoji, a skin-tone modifier, flag
/// sequences, and rare 4-byte CJK Extension-B characters) -- digital
/// round-trip plus a moderate-noise round-trip, to confirm the charset
/// layer's escaping/resync logic holds up under both clean and lossy
/// conditions, not just clean ones.
#[test]
fn extreme_utf8_survives_clean_and_moderate_noise() {
    let text = "Hello 你好 مرحبا Привет e\u{0301} $€¥ ∑∞ 👨‍👩‍👧‍👦 👍🏽 🇺🇸🇯🇵 𠀀\u{200D}";
    let audio_clean = build_and_modulate(text, 20);
    assert_eq!(decode_audio(&audio_clean, 20).as_deref(), Some(text));

    for seed in 0..5 {
        let mut audio = build_and_modulate(text, 20);
        awgn(&mut audio, 15.0, seed);
        let result = decode_audio(&audio, 20);
        assert!(
            result.is_none() || result.as_deref() == Some(text),
            "extreme UTF-8 case returned wrong text at seed={seed}: {result:?}"
        );
    }
}
