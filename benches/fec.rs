//! Benchmarks for the hand-rolled Reed-Solomon codec ([`textovervoice_core::fec`]).
//! Correctness of this module got the most scrutiny during the port (see
//! its module docs); this covers the other axis -- is the hand-rolled
//! implementation fast enough not to matter in practice, at realistic
//! payload sizes (single-chunk, ~245-byte chunk boundary, and multi-chunk).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use textovervoice_core::fec;

fn bench_protect(c: &mut Criterion) {
    let mut group = c.benchmark_group("fec_protect");
    for &parity_bytes in &[10usize, 20, 40] {
        for &data_len in &[20usize, 245, 1000] {
            let data = vec![0xABu8; data_len];
            group.throughput(Throughput::Bytes(data_len as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("parity{parity_bytes}"), data_len),
                &data,
                |b, data| b.iter(|| fec::protect(std::hint::black_box(data), parity_bytes)),
            );
        }
    }
    group.finish();
}

fn bench_recover_clean(c: &mut Criterion) {
    let mut group = c.benchmark_group("fec_recover_clean");
    for &parity_bytes in &[10usize, 20, 40] {
        for &data_len in &[20usize, 245, 1000] {
            let data = vec![0xABu8; data_len];
            let parity = fec::protect(&data, parity_bytes);
            group.throughput(Throughput::Bytes(data_len as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("parity{parity_bytes}"), data_len),
                &(data, parity),
                |b, (data, parity)| {
                    b.iter(|| fec::recover(std::hint::black_box(data), parity, parity_bytes))
                },
            );
        }
    }
    group.finish();
}

/// The more expensive path: recovering with actual errors present
/// (exercises Berlekamp-Massey/Chien/Forney, not just the "syndromes are
/// all zero, return early" fast path that `bench_recover_clean` mostly
/// hits for a truly clean channel).
fn bench_recover_with_errors(c: &mut Criterion) {
    let mut group = c.benchmark_group("fec_recover_with_max_correctable_errors");
    for &parity_bytes in &[10usize, 20, 40] {
        let data = vec![0xABu8; 100];
        let parity = fec::protect(&data, parity_bytes);
        let mut codeword = data.clone();
        codeword.extend_from_slice(&parity);
        let t = parity_bytes / 2;
        for i in 0..t {
            codeword[i * 2] ^= 0xFF;
        }
        let (corrupted_data, corrupted_parity) = codeword.split_at(data.len());
        let corrupted_data = corrupted_data.to_vec();
        let corrupted_parity = corrupted_parity.to_vec();

        group.bench_with_input(
            BenchmarkId::new("parity", parity_bytes),
            &(corrupted_data, corrupted_parity),
            |b, (d, p)| b.iter(|| fec::recover(std::hint::black_box(d), p, parity_bytes)),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_protect,
    bench_recover_clean,
    bench_recover_with_errors
);
criterion_main!(benches);
