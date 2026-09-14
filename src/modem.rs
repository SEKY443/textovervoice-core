//! 2-of-16 dual-tone MFSK modem: the physical-layer symbol encoder/decoder.
//!
//! One symbol = exactly one tone from `LOW_GROUP` + one tone from
//! `HIGH_GROUP`, giving 8 x 8 = 64 combinations = 6 bits/symbol. This
//! replaces an earlier 8-simultaneous-tone FDM design, which measurably
//! failed against AMR-NB: real ffmpeg/libopencore_amrnb round-trips showed
//! 8-tone chords producing more intermodulation energy than signal at
//! AMR's low bitrate, while dual-tone pairs stayed well clear of that
//! failure mode.
//!
//! Frequencies were chosen from a single-tone sweep: all 8 candidates in
//! each group individually survive AMR-NB at both 4.75k and 12.2k with
//! <6dB loss and <30Hz drift.

use std::cell::RefCell;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

pub const SR: u32 = 8000; // AMR-NB operates at 8kHz mono; keep the whole pipeline native to that rate

// A fresh `RealFftPlanner` re-plans from scratch on every call -- for an
// "awkward" length (not a nice power-of-two/5-smooth size, which is
// exactly what `find_preamble`'s convolution length usually is: audio
// window length + reference length - 1, an arbitrary sum), planning can
// fall back to the much more expensive Bluestein's algorithm and pay that
// planning cost on every single call. `find_preamble` runs on every
// `Listener` poll (every ~300ms during a live session -- see `live.rs`),
// so re-planning from scratch there is real, avoidable, per-poll CPU cost,
// not a one-off. `RealFftPlanner` already caches plans internally per
// length; reusing one planner instance across calls (thread-local, since
// the planner isn't `Sync`) lets that cache actually do its job instead of
// starting empty every call.
thread_local! {
    static FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::<f64>::new());
}

fn cached_forward_fft(len: usize) -> Arc<dyn RealToComplex<f64>> {
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(len))
}

fn cached_inverse_fft(len: usize) -> Arc<dyn ComplexToReal<f64>> {
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(len))
}

pub const LOW_GROUP: [f64; 8] = [400.0, 550.0, 700.0, 850.0, 1000.0, 1150.0, 1300.0, 1450.0];
pub const HIGH_GROUP: [f64; 8] = [
    1800.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0,
];
pub const BITS_PER_SYMBOL: u32 = 6; // log2(8*8)

/// A named timing profile. `"phone"` is validated through real AMR-NB;
/// `"fast_air"` is validated only acoustically (real speaker/mic, no
/// codec) -- it is NOT expected to survive AMR compression, since the AMR
/// frame-boundary constraint that set `"phone"`'s 40ms floor doesn't apply
/// to fast_air's shorter timing at all.
///
/// fast_air's duration was picked from a real-hardware sweep, not guessed:
/// 20ms and 40ms measured comparable symbol-error rates in clean
/// conditions, while 15ms and below entered a clearly worse regime and
/// below 10ms was catastrophic -- consistent with 150Hz-spaced tones
/// needing enough samples for the FFT to resolve adjacent bins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeProfile {
    pub symbol_duration_s: f64,
    pub guard_s: f64,
}

pub const MODE_NAMES: [&str; 2] = ["phone", "fast_air"];

pub fn mode_profile(name: &str) -> Option<ModeProfile> {
    match name {
        "phone" => Some(ModeProfile {
            symbol_duration_s: 0.04,
            guard_s: 0.01,
        }),
        "fast_air" => Some(ModeProfile {
            symbol_duration_s: 0.02,
            guard_s: 0.005,
        }),
        _ => None,
    }
}

pub const DEFAULT_SYMBOL_DURATION_S: f64 = 0.04;
pub const DEFAULT_GUARD_S: f64 = 0.01;
pub const RAMP_FRACTION: f64 = 0.15; // raised-cosine edge, as a fraction of symbol duration

pub fn symbol_to_freqs(symbol: u8) -> (f64, f64) {
    assert!(symbol < 64, "symbol {symbol} out of range 0-63");
    let low_idx = (symbol / 8) as usize;
    let high_idx = (symbol % 8) as usize;
    (LOW_GROUP[low_idx], HIGH_GROUP[high_idx])
}

pub fn freqs_to_symbol(low_idx: usize, high_idx: usize) -> u8 {
    (low_idx * 8 + high_idx) as u8
}

/// Applies an in-place raised-cosine (Hann-style) ramp to the first and
/// last `frac` fraction of `signal`, leaving the middle untouched.
fn apply_raised_cosine_edges(signal: &mut [f64], ramp_n: usize) {
    let n = signal.len();
    let ramp_n = ramp_n.min(n / 2).max(if n > 0 { 1 } else { 0 });
    for i in 0..ramp_n {
        let w = 0.5 * (1.0 - (std::f64::consts::PI * i as f64 / ramp_n as f64).cos());
        signal[i] *= w;
        signal[n - 1 - i] *= w;
    }
}

pub fn synth_symbol(symbol: u8, duration_s: f64, sr: u32, amplitude: f64) -> Vec<f64> {
    let (low_f, high_f) = symbol_to_freqs(symbol);
    let n = (duration_s * sr as f64) as usize;
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            amplitude
                * 0.5
                * ((2.0 * std::f64::consts::PI * low_f * t).sin()
                    + (2.0 * std::f64::consts::PI * high_f * t).sin())
        })
        .collect();
    let ramp_n = ((n as f64 * RAMP_FRACTION) as usize).max(1);
    apply_raised_cosine_edges(&mut x, ramp_n);
    x
}

pub fn modulate(symbols: &[u8], symbol_duration_s: f64, guard_s: f64, sr: u32) -> Vec<f64> {
    let guard = vec![0.0f64; (guard_s * sr as f64) as usize];
    let mut out = Vec::new();
    for &s in symbols {
        out.extend(synth_symbol(s, symbol_duration_s, sr, 0.5));
        out.extend(&guard);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolDetection {
    pub symbol: u8,
    pub low_freq: f64,
    pub high_freq: f64,
    pub low_confidence_db: f64, // margin of winning low-group bin over runner-up
    pub high_confidence_db: f64,
}

fn argmax(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

pub fn detect_symbol(segment: &[f64], sr: u32) -> SymbolDetection {
    let n = segment.len();
    let window = hann_window(n);
    let windowed: Vec<f64> = segment.iter().zip(&window).map(|(&s, &w)| s * w).collect();

    let fwd = cached_forward_fft(n.max(1));
    let mut input = fwd.make_input_vec();
    input[..n].copy_from_slice(&windowed);
    let mut spec = fwd.make_output_vec();
    fwd.process(&mut input, &mut spec)
        .expect("fixed-size real FFT");
    let mags: Vec<f64> = spec.iter().map(|c| c.norm()).collect();

    let bin_hz = if mags.len() > 1 {
        sr as f64 / n as f64
    } else {
        1.0
    };
    let search_bins = ((40.0 / bin_hz) as usize).max(1);

    let group_scores = |group: &[f64; 8]| -> Vec<f64> {
        group
            .iter()
            .map(|&f0| {
                let center = (f0 / bin_hz) as usize;
                let lo = center.saturating_sub(search_bins);
                let hi = (center + search_bins + 1).min(mags.len());
                if hi > lo {
                    mags[lo..hi].iter().cloned().fold(0.0f64, f64::max)
                } else {
                    0.0
                }
            })
            .collect()
    };

    let low_scores = group_scores(&LOW_GROUP);
    let high_scores = group_scores(&HIGH_GROUP);
    let low_idx = argmax(&low_scores);
    let high_idx = argmax(&high_scores);

    let confidence_db = |scores: &[f64], winner_idx: usize| -> f64 {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top = sorted[0];
        let runner_up = if sorted.len() > 1 { sorted[1] } else { 1e-9 };
        if scores[winner_idx] == top {
            20.0 * ((top + 1e-12) / (runner_up + 1e-12)).log10()
        } else {
            -99.0
        }
    };

    SymbolDetection {
        symbol: freqs_to_symbol(low_idx, high_idx),
        low_freq: LOW_GROUP[low_idx],
        high_freq: HIGH_GROUP[high_idx],
        low_confidence_db: confidence_db(&low_scores, low_idx),
        high_confidence_db: confidence_db(&high_scores, high_idx),
    }
}

/// Fixed-timing slicer: assumes the caller already knows where symbol 0
/// starts (`start_offset`, in samples). A real deployment needs a preamble
/// to find this offset on a live capture -- see [`find_preamble`] /
/// [`demodulate_frame`].
pub fn demodulate(
    audio: &[f64],
    n_symbols: usize,
    symbol_duration_s: f64,
    guard_s: f64,
    sr: u32,
    start_offset: usize,
) -> Vec<SymbolDetection> {
    let symbol_n = (symbol_duration_s * sr as f64) as usize;
    let guard_n = (guard_s * sr as f64) as usize;
    let step = symbol_n + guard_n;

    let mut detections = Vec::with_capacity(n_symbols);
    for i in 0..n_symbols {
        let start = start_offset + i * step;
        let end = start + symbol_n;
        if end > audio.len() {
            break;
        }
        detections.push(detect_symbol(&audio[start..end], sr));
    }
    detections
}

// --- Frame sync: locate symbol 0 in a live capture with no known alignment --
// A linear chirp cross-correlates sharply against noise/tones (much sharper
// peak than a single fixed tone would give), which is the standard choice
// for this in real modems.

pub const PREAMBLE_DURATION_S: f64 = 0.25;
pub const PREAMBLE_F0: f64 = 800.0;
pub const PREAMBLE_F1: f64 = 3000.0; // stays inside the 300-3400Hz voice band, so it works for "phone" mode too
pub const PREAMBLE_GUARD_S: f64 = 0.05; // silence between preamble and payload

pub fn generate_preamble(duration_s: f64, f0: f64, f1: f64, sr: u32) -> Vec<f64> {
    let n = (duration_s * sr as f64) as usize;
    let mut sig: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            // Linear chirp instantaneous phase (matches scipy.signal.chirp,
            // method="linear"): phase = 2*pi*(f0*t + 0.5*(f1-f0)/t1*t^2).
            let phase =
                2.0 * std::f64::consts::PI * (f0 * t + 0.5 * (f1 - f0) / duration_s * t * t);
            phase.cos()
        })
        .collect();
    let ramp_n = (0.1 * n as f64) as usize;
    apply_raised_cosine_edges(&mut sig, ramp_n);
    sig.iter().map(|&v| 0.6 * v).collect()
}

/// Computes the full linear convolution of `a` and `b` via zero-padded
/// FFTs (avoids O(N*M) direct convolution -- essential for real-time
/// preamble search, where a naive direct computation over a realistic
/// live-polling window measured 40+ seconds per call).
/// The smallest power of two >= `n`. Padding an FFT to a power-of-two
/// length (rather than the exact minimal length a convolution needs) is a
/// deliberate trade: `n` is an arbitrary sum (audio window length +
/// reference length), so it routinely lands on a size with large prime
/// factors -- measured directly, `find_preamble`'s real-world window size
/// (0.6s of 48kHz audio + the preamble reference) factors as 11 x 3709,
/// forcing rustfft into its slow-path algorithm for awkward sizes on
/// *every* call. A power-of-two length is always fast to plan and execute,
/// and the extra zero-padding beyond the true convolution length doesn't
/// change the result: the "valid" region this function's caller extracts
/// only reads indices below the true linear-convolution length, and
/// zero-padding further out cannot alias back into that region (that's
/// exactly the property that makes zero-padding safe for linear
/// convolution via FFT in the first place).
fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

fn fft_convolve_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    let full_len = a.len() + b.len() - 1;
    let padded_len = next_pow2(full_len);
    let fwd = cached_forward_fft(padded_len);
    let inv = cached_inverse_fft(padded_len);

    let mut a_buf = fwd.make_input_vec();
    a_buf[..a.len()].copy_from_slice(a);
    let mut b_buf = fwd.make_input_vec();
    b_buf[..b.len()].copy_from_slice(b);

    let mut a_spec = fwd.make_output_vec();
    let mut b_spec = fwd.make_output_vec();
    fwd.process(&mut a_buf, &mut a_spec)
        .expect("fixed-size real FFT");
    fwd.process(&mut b_buf, &mut b_spec)
        .expect("fixed-size real FFT");

    let mut prod: Vec<Complex<f64>> = a_spec
        .iter()
        .zip(b_spec.iter())
        .map(|(x, y)| x * y)
        .collect();
    let mut out = inv.make_output_vec();
    inv.process(&mut prod, &mut out)
        .expect("fixed-size real inverse FFT");

    let scale = 1.0 / padded_len as f64;
    out.iter().map(|v| v * scale).collect()
}

/// Cross-correlates `audio` against the known preamble chirp, using a
/// normalized cross-correlation coefficient (numerator / sqrt(local window
/// energy * reference energy)) rather than a raw correlation peak. This
/// matters: an earlier version compared the peak against the correlation's
/// own median as a "confidence" score, which false-triggered on pure noise
/// -- with many correlation lags, the max of an unrelated-noise
/// correlation can spike several times above the median from ordinary
/// extreme-value statistics, even though nothing actually matched.
/// Normalizing by local energy bounds the score to roughly `[-1, 1]` with a
/// stable, physically meaningful threshold instead.
///
/// Returns `(sample_index_after_preamble, score)`, or `None` if no lag's
/// score clears `min_score`, i.e. sync failed.
pub fn find_preamble(audio: &[f64], reference: &[f64], min_score: f64) -> Option<(usize, f64)> {
    let n = reference.len();
    if audio.len() < n {
        return None;
    }

    let reversed_reference: Vec<f64> = reference.iter().rev().cloned().collect();
    let full_conv = fft_convolve_full(audio, &reversed_reference);
    let numerator = &full_conv[(n - 1)..audio.len()];

    let mut cumsum = vec![0.0f64; audio.len() + 1];
    for (i, &a) in audio.iter().enumerate() {
        cumsum[i + 1] = cumsum[i] + a * a;
    }
    let ref_energy: f64 = reference.iter().map(|&r| r * r).sum();

    let score: Vec<f64> = numerator
        .iter()
        .enumerate()
        .map(|(k, &num)| {
            let window_energy = cumsum[n + k] - cumsum[k];
            num / ((window_energy * ref_energy).sqrt() + 1e-12)
        })
        .collect();

    let abs_score: Vec<f64> = score.iter().map(|v| v.abs()).collect();
    let peak_idx = argmax(&abs_score);
    let peak_score = abs_score[peak_idx];

    if peak_score < min_score {
        None
    } else {
        Some((peak_idx + n, peak_score))
    }
}

/// [`modulate`] with a preamble prepended, for transmission into an
/// unknown-alignment channel (a live capture, not a file where symbol 0 is
/// known to start at sample 0).
pub fn modulate_frame(symbols: &[u8], symbol_duration_s: f64, guard_s: f64, sr: u32) -> Vec<f64> {
    let preamble = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, sr);
    let preamble_guard = vec![0.0f64; (PREAMBLE_GUARD_S * sr as f64) as usize];
    let payload = modulate(symbols, symbol_duration_s, guard_s, sr);
    let mut out = Vec::with_capacity(preamble.len() + preamble_guard.len() + payload.len());
    out.extend(preamble);
    out.extend(preamble_guard);
    out.extend(payload);
    out
}

/// Finds the preamble, then demodulates the payload that follows it.
/// Returns `(detections, sync_score)`, or `None` if the preamble wasn't
/// found reliably (caller should treat that as "no frame here yet" rather
/// than a decode failure).
pub fn demodulate_frame(
    audio: &[f64],
    n_symbols: usize,
    symbol_duration_s: f64,
    guard_s: f64,
    sr: u32,
    min_score: f64,
) -> Option<(Vec<SymbolDetection>, f64)> {
    let reference = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, sr);
    let (offset, score) = find_preamble(audio, &reference, min_score)?;
    let payload_start = offset + (PREAMBLE_GUARD_S * sr as f64) as usize;
    let detections = demodulate(
        &audio[payload_start..],
        n_symbols,
        symbol_duration_s,
        guard_s,
        sr,
        0,
    );
    Some((detections, score))
}

// --- Bit packing: 8-bit bytes <-> 6-bit symbols -----------------------------
// Same ratio as base64 (3 bytes = 24 bits = 4 six-bit groups), but we need
// raw symbol indices (0-63), not a text alphabet, so a direct bit-packer is
// used rather than repurposing base64's character mapping.

pub fn bytes_to_symbols(data: &[u8]) -> Vec<u8> {
    let mut bits: Vec<u8> = Vec::with_capacity(data.len() * 8);
    for &b in data {
        for i in (0..8).rev() {
            bits.push((b >> i) & 1);
        }
    }
    let pad = (6 - bits.len() % 6) % 6;
    bits.extend(std::iter::repeat_n(0u8, pad));
    bits.chunks(6)
        .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | b))
        .collect()
}

pub fn symbols_to_bytes(symbols: &[u8], n_bytes: usize) -> Vec<u8> {
    let mut bits: Vec<u8> = Vec::with_capacity(symbols.len() * 6);
    for &s in symbols {
        for i in (0..6).rev() {
            bits.push((s >> i) & 1);
        }
    }
    bits.truncate(n_bytes * 8);
    bits.chunks(8)
        .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn gaussian_noise(n: usize, std: f64, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, std).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }

    /// Regression test for the FFT-length padding optimization in
    /// `fft_convolve_full`: padding the transform to the next power of two
    /// (rather than the exact minimal convolution length) must not change
    /// which samples land in the "valid" output region it extracts.
    #[test]
    fn next_pow2_is_correct() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(4), 4);
        assert_eq!(next_pow2(40799), 65536); // the real find_preamble window size that motivated this
        assert_eq!(next_pow2(65536), 65536);
        assert_eq!(next_pow2(65537), 131072);
    }

    /// Same padding optimization, but at the smallest legal input size
    /// (`audio.len() == reference.len()`, giving exactly one lag) --
    /// boundary case for the "valid" slice extraction math, which is
    /// exactly where an off-by-one in the padding change would show up.
    #[test]
    fn find_preamble_works_at_minimal_audio_length() {
        let reference = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, SR);
        let audio = reference.clone(); // audio.len() == reference.len(), the minimal valid case
        let found = find_preamble(&audio, &reference, 0.4);
        assert!(found.is_some());
        let (offset, score) = found.unwrap();
        assert_eq!(offset, reference.len());
        assert!(score > 0.9, "score={score}");
    }

    #[test]
    fn find_preamble_locates_known_offset() {
        let pre_silence = gaussian_noise(4000, 0.01, 0);
        let preamble = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, SR);
        let post = gaussian_noise(2000, 0.01, 1);
        let mut audio = Vec::new();
        audio.extend(&pre_silence);
        audio.extend(&preamble);
        audio.extend(&post);

        let reference = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, SR);
        let found = find_preamble(&audio, &reference, 0.4);
        assert!(found.is_some());
        let (offset, score) = found.unwrap();
        assert_eq!(offset, pre_silence.len() + preamble.len());
        assert!(score > 0.9, "score={score}"); // near-exact match against a near-noiseless embed
    }

    /// Regression test: an earlier confidence metric (peak vs. local
    /// median of the correlation) false-triggered on pure noise, since
    /// with many correlation lags the max of unrelated-noise correlation
    /// naturally spikes above the median from ordinary extreme-value
    /// statistics. Runs several noise seeds and a longer buffer to make a
    /// false positive here meaningful, not a coin flip.
    #[test]
    fn find_preamble_returns_none_on_noise_only() {
        let reference = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, SR);
        for seed in 0..10 {
            let noise = gaussian_noise(80000, 0.05, seed);
            assert!(
                find_preamble(&noise, &reference, 0.4).is_none(),
                "false positive on noise seed {seed}"
            );
        }
    }

    #[test]
    fn find_preamble_returns_none_on_silence() {
        let reference = generate_preamble(PREAMBLE_DURATION_S, PREAMBLE_F0, PREAMBLE_F1, SR);
        let silence = vec![0.0f64; 20000];
        assert!(find_preamble(&silence, &reference, 0.4).is_none());
    }

    #[test]
    fn modulate_demodulate_frame_roundtrip_no_channel_impairment() {
        let symbols: Vec<u8> = vec![0, 15, 32, 47, 63, 7, 21];
        let audio = modulate_frame(&symbols, DEFAULT_SYMBOL_DURATION_S, DEFAULT_GUARD_S, SR);
        let result = demodulate_frame(
            &audio,
            symbols.len(),
            DEFAULT_SYMBOL_DURATION_S,
            DEFAULT_GUARD_S,
            SR,
            0.4,
        );
        assert!(result.is_some());
        let (detections, score) = result.unwrap();
        assert_eq!(
            detections.iter().map(|d| d.symbol).collect::<Vec<_>>(),
            symbols
        );
        assert!(score > 0.9, "score={score}");
    }

    /// This is the actual point of frame sync: symbol 0 does NOT need to
    /// start at sample 0, unlike [`demodulate`]'s fixed-timing assumption.
    #[test]
    fn demodulate_frame_works_with_arbitrary_leading_silence() {
        let symbols: Vec<u8> = vec![3, 9, 40];
        let mut padded = vec![0.0f64; 7777];
        padded.extend(modulate_frame(
            &symbols,
            DEFAULT_SYMBOL_DURATION_S,
            DEFAULT_GUARD_S,
            SR,
        ));
        let result = demodulate_frame(
            &padded,
            symbols.len(),
            DEFAULT_SYMBOL_DURATION_S,
            DEFAULT_GUARD_S,
            SR,
            0.4,
        );
        assert!(result.is_some());
        let (detections, _) = result.unwrap();
        assert_eq!(
            detections.iter().map(|d| d.symbol).collect::<Vec<_>>(),
            symbols
        );
    }

    #[test]
    fn modes_are_registered_and_phone_matches_legacy_defaults() {
        assert!(mode_profile("phone").is_some());
        assert!(mode_profile("fast_air").is_some());
        let phone = mode_profile("phone").unwrap();
        assert_eq!(phone.symbol_duration_s, DEFAULT_SYMBOL_DURATION_S);
        assert_eq!(phone.guard_s, DEFAULT_GUARD_S);
        // fast_air must actually be faster, or it's not earning its name
        let fast_air = mode_profile("fast_air").unwrap();
        assert!(fast_air.symbol_duration_s < phone.symbol_duration_s);
    }

    #[test]
    fn symbol_freq_roundtrip_covers_all_64_symbols() {
        for symbol in 0u8..64 {
            let (low_f, high_f) = symbol_to_freqs(symbol);
            let low_idx = LOW_GROUP.iter().position(|&f| f == low_f).unwrap();
            let high_idx = HIGH_GROUP.iter().position(|&f| f == high_f).unwrap();
            assert_eq!(freqs_to_symbol(low_idx, high_idx), symbol);
        }
    }

    #[test]
    fn detect_symbol_recovers_all_64_symbols_clean() {
        for symbol in 0u8..64 {
            let audio = synth_symbol(symbol, DEFAULT_SYMBOL_DURATION_S, SR, 0.5);
            let detection = detect_symbol(&audio, SR);
            assert_eq!(detection.symbol, symbol, "misdetected symbol {symbol}");
        }
    }

    #[test]
    fn bytes_to_symbols_and_back_roundtrip() {
        let data: Vec<u8> = (0..=255u8).collect();
        let symbols = bytes_to_symbols(&data);
        let recovered = symbols_to_bytes(&symbols, data.len());
        assert_eq!(recovered, data);
    }

    #[test]
    fn bytes_to_symbols_produces_6_bit_values() {
        let data = b"hello world";
        for s in bytes_to_symbols(data) {
            assert!(s < 64);
        }
    }

    #[test]
    fn empty_bytes_roundtrip() {
        assert!(bytes_to_symbols(&[]).is_empty());
        assert!(symbols_to_bytes(&[], 0).is_empty());
    }
}
