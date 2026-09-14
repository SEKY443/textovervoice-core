# textovervoice-core

Portable protocol/DSP/FEC core for [TextOverVoice](https://github.com/SEKY443/CLI-TextOverVoice)
(a text-over-audio modem for phone calls / VoIP), extracted so it can be
compiled to WebAssembly for the browser-based client at
[TOVChat.github.io](https://github.com/SEKY443/TOVChat.github.io).

## What's here

Pure Rust, no OS-specific I/O:

- `modem` — MFSK modulator/demodulator, FFT-based preamble sync (`realfft`/`rustfft`)
- `fec` — hand-rolled Reed-Solomon + CRC-16
- `framing`, `charset`, `dictionary`, `codes` — byte-stuffing, UTF-8 boundary
  encoding, dictionary compression
- `protocol` — frame build/parse (`Legacy` and `ProtectedHeader` wire
  formats), addressing, SRC_ID
- `message` — multi-frame splitting/reassembly
- `crypto` — X25519 + ChaCha20-Poly1305 + HKDF-SHA256

Verified to build cleanly for `wasm32-unknown-unknown` (see the
`wasm-browser` feature below) with no source changes beyond what's already
here.

## Relationship to CLI-TextOverVoice

This is a **copy**, not a shared dependency: the
[CLI-TextOverVoice](https://github.com/SEKY443/CLI-TextOverVoice) repo keeps
its own copy of these modules and is not restructured or otherwise touched
by this split. That was a deliberate choice to avoid any risk to the
existing, working CLI while the web client is being built — the tradeoff is
that the two copies can drift out of sync if either is bugfixed
independently. If that becomes a real maintenance problem, the fix is for
CLI-TextOverVoice to switch to depending on this crate instead of its own
copy.

## `wasm-browser` feature

`getrandom` needs an explicit backend to produce randomness on
`wasm32-unknown-unknown` (browsers have no OS RNG). This crate exposes that
as an opt-in feature rather than forcing it on every consumer:

```toml
textovervoice-core = { git = "https://github.com/SEKY443/textovervoice-core", features = ["wasm-browser"] }
```

Enable it when building for the browser (routes through
`crypto.getRandomValues`); leave it off for any native build.

## License

MIT, see [LICENSE](LICENSE).
