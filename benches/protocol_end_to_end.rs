//! End-to-end benchmarks: build_message+modulate (the `encode` path) and
//! find_preamble+demodulate+parse_frame (the `decode`/live-listen path),
//! at the same message lengths the original Python prototype's README
//! throughput table used (19/68/251 chars) so the two are easy to line up
//! conceptually -- though note what's actually being measured differs:
//! the original table reports *protocol* throughput (chars per second of
//! *transmitted audio*, a property of the wire timing, unchanged by this
//! port since the timing profiles were copied exactly). What matters for a
//! Rust rewrite is *CPU* throughput -- how many real-time seconds of audio
//! this implementation can process per CPU-second, which is the actual
//! thing a faster/slower implementation could change. Each benchmark
//! prints a one-time "Nx real-time" figure computed from the modulated
//! audio's duration, alongside criterion's own timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use textovervoice_core::{fec, message, modem};

const SAMPLE_TEXTS: [(&str, &str); 3] = [
    ("19chars", "Hello, how are you?"),
    ("68chars", "The government should provide information about something."),
    (
        "251chars",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once, which makes it useful for testing fonts, keyboards, and text encoding systems across many different applications and platforms today.",
    ),
];

fn build_audio(
    text: &str,
    mode: &str,
    parity_bytes: usize,
    session_key: Option<&[u8; 32]>,
) -> (Vec<f64>, usize) {
    let profile = modem::mode_profile(mode).unwrap();
    let frames = message::build_message(
        text,
        &message::MessageBuildOptions {
            parity_bytes,
            use_dictionary: true,
            session_key,
            ..Default::default()
        },
    )
    .unwrap();
    let mut audio = Vec::new();
    for f in &frames {
        let symbols = modem::bytes_to_symbols(f);
        audio.extend(modem::modulate_frame(
            &symbols,
            profile.symbol_duration_s,
            profile.guard_s,
            modem::SR,
        ));
    }
    (audio, frames.len())
}

fn bench_encode_path(c: &mut Criterion) {
    for mode in ["phone", "fast_air"] {
        let mut group = c.benchmark_group(format!("encode_path_{mode}"));
        for (name, text) in SAMPLE_TEXTS {
            let (audio, _frames) = build_audio(text, mode, fec::DEFAULT_PARITY_BYTES, None);
            let real_time_s = audio.len() as f64 / modem::SR as f64;
            eprintln!("[{mode}/{name}] audio duration = {real_time_s:.2}s (informational, not a criterion metric)");

            group.bench_with_input(
                BenchmarkId::new("plain_dictionary", name),
                text,
                |b, text| {
                    b.iter(|| {
                        build_audio(
                            std::hint::black_box(text),
                            mode,
                            fec::DEFAULT_PARITY_BYTES,
                            None,
                        )
                    });
                },
            );
        }
        group.finish();
    }
}

fn bench_decode_path(c: &mut Criterion) {
    for mode in ["phone", "fast_air"] {
        let profile = modem::mode_profile(mode).unwrap();
        let mut group = c.benchmark_group(format!("decode_path_{mode}"));
        for (name, text) in SAMPLE_TEXTS {
            let (audio, _frames) = build_audio(text, mode, fec::DEFAULT_PARITY_BYTES, None);
            let real_time_s = audio.len() as f64 / modem::SR as f64;

            group.bench_with_input(
                BenchmarkId::new("plain_dictionary", name),
                &audio,
                |b, audio| {
                    b.iter(|| {
                        let reference = modem::generate_preamble(
                            modem::PREAMBLE_DURATION_S,
                            modem::PREAMBLE_F0,
                            modem::PREAMBLE_F1,
                            modem::SR,
                        );
                        let (offset, _score) =
                            modem::find_preamble(std::hint::black_box(audio), &reference, 0.4)
                                .unwrap();
                        let payload_start =
                            offset + (modem::PREAMBLE_GUARD_S * modem::SR as f64) as usize;
                        let step_n = ((profile.symbol_duration_s + profile.guard_s)
                            * modem::SR as f64) as usize;
                        let n_symbols = ((audio.len() - payload_start) / step_n).min(2000);
                        let detections = modem::demodulate(
                            &audio[payload_start..],
                            n_symbols,
                            profile.symbol_duration_s,
                            profile.guard_s,
                            modem::SR,
                            0,
                        );
                        let symbols: Vec<u8> = detections.iter().map(|d| d.symbol).collect();
                        let n_bytes = (symbols.len() * modem::BITS_PER_SYMBOL as usize) / 8;
                        let frame_codes = modem::symbols_to_bytes(&symbols, n_bytes);
                        textovervoice_core::protocol::parse_frame(
                            &frame_codes,
                            fec::DEFAULT_PARITY_BYTES,
                            true,
                            None,
                            None,
                        )
                    });
                },
            );
            let _ = real_time_s; // printed above alongside encode path; decode's own factor is derived from criterion's report
        }
        group.finish();
    }
}

criterion_group!(benches, bench_encode_path, bench_decode_path);
criterion_main!(benches);
