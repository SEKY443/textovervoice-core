//! Benchmarks for the DSP core: symbol synthesis/detection and preamble
//! search. These matter for two different reasons:
//!
//! - `synth_symbol`/`detect_symbol` cost bounds how fast `encode`/`decode`
//!   can chew through a file, independent of real-time audio playback.
//! - `find_preamble` cost bounds how much CPU one `Listener` poll cycle
//!   burns during live `listen`/`chat` -- it runs every `POLL_INTERVAL`
//!   (300ms) for the whole session, so it needs to stay cheap relative to
//!   that budget, not just "fast in absolute terms." This is exactly the
//!   category of cost the original prototype's O(N^2) buffer-growth bug
//!   lived in (see `live.rs`'s docs) -- unlike that bug, this one isn't
//!   about correctness, but a regression here would silently erode the
//!   real-time headroom a live session depends on.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use textovervoice_core::modem;

fn bench_synth_symbol(c: &mut Criterion) {
    let mut group = c.benchmark_group("synth_symbol");
    for &sr in &[modem::SR, 48_000] {
        group.bench_with_input(BenchmarkId::new("phone_mode", sr), &sr, |b, &sr| {
            b.iter(|| {
                modem::synth_symbol(
                    std::hint::black_box(37),
                    modem::DEFAULT_SYMBOL_DURATION_S,
                    sr,
                    0.5,
                )
            });
        });
    }
    group.finish();
}

fn bench_detect_symbol(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect_symbol");
    for &sr in &[modem::SR, 48_000] {
        let segment = modem::synth_symbol(37, modem::DEFAULT_SYMBOL_DURATION_S, sr, 0.5);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("phone_mode", sr),
            &segment,
            |b, segment| {
                b.iter(|| modem::detect_symbol(std::hint::black_box(segment), sr));
            },
        );
    }
    group.finish();
}

/// End-to-end modulate+demodulate for a realistic 64-symbol frame, at
/// DEVICE_SR (48kHz) -- the rate live audio actually runs at.
fn bench_modulate_demodulate_frame(c: &mut Criterion) {
    let symbols: Vec<u8> = (0u8..64).collect();
    let mut group = c.benchmark_group("modulate_demodulate_frame_64symbols_48k");

    group.bench_function("modulate", |b| {
        b.iter(|| {
            modem::modulate_frame(
                std::hint::black_box(&symbols),
                modem::DEFAULT_SYMBOL_DURATION_S,
                modem::DEFAULT_GUARD_S,
                48_000,
            )
        });
    });

    let audio = modem::modulate_frame(
        &symbols,
        modem::DEFAULT_SYMBOL_DURATION_S,
        modem::DEFAULT_GUARD_S,
        48_000,
    );
    group.bench_function("demodulate_frame", |b| {
        b.iter(|| {
            modem::demodulate_frame(
                std::hint::black_box(&audio),
                symbols.len(),
                modem::DEFAULT_SYMBOL_DURATION_S,
                modem::DEFAULT_GUARD_S,
                48_000,
                0.4,
            )
        });
    });
    group.finish();
}

/// The cost that matters for live listening: one `find_preamble` call over
/// a realistic per-poll search window (`SEARCH_WINDOW_S` = 0.6s at 48kHz in
/// `live.rs`), both when a preamble is actually present and when the
/// window is just silence/noise (the much more common case during a real
/// session -- most polls find nothing).
fn bench_find_preamble_search_window(c: &mut Criterion) {
    let reference = modem::generate_preamble(
        modem::PREAMBLE_DURATION_S,
        modem::PREAMBLE_F0,
        modem::PREAMBLE_F1,
        48_000,
    );
    let window_n = (0.6 * 48_000.0) as usize;

    let mut group = c.benchmark_group("find_preamble_0.6s_window_48k");

    let silence = vec![0.0f64; window_n];
    group.bench_function("silence", |b| {
        b.iter(|| modem::find_preamble(std::hint::black_box(&silence), &reference, 0.4));
    });

    let mut with_preamble = vec![0.0f64; window_n];
    let pre = modem::generate_preamble(
        modem::PREAMBLE_DURATION_S,
        modem::PREAMBLE_F0,
        modem::PREAMBLE_F1,
        48_000,
    );
    with_preamble[1000..1000 + pre.len()].copy_from_slice(&pre);
    group.bench_function("preamble_present", |b| {
        b.iter(|| modem::find_preamble(std::hint::black_box(&with_preamble), &reference, 0.4));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_synth_symbol,
    bench_detect_symbol,
    bench_modulate_demodulate_frame,
    bench_find_preamble_search_window
);
criterion_main!(benches);
