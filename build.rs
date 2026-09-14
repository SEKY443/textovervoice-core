//! Generates two compile-time perfect-hash maps (word -> 3-letter code, and
//! its inverse) from `data/dict_en.json`, so the dictionary module needs no
//! runtime file I/O or JSON parsing -- see `tools/build_dictionary.py` in
//! the original Python prototype for how that data file itself is produced
//! (not ported: it's a dev-only tool built on `wordfreq`, not a runtime
//! dependency, same as in the original project).

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() {
    let dict_path = "data/dict_en.json";
    println!("cargo:rerun-if-changed={dict_path}");

    let raw = std::fs::read_to_string(dict_path).expect("read data/dict_en.json");
    let table = parse_json_string_map(&raw);

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("dictionary_data.rs");
    let mut out = BufWriter::new(File::create(&dest_path).unwrap());

    let code_literals: Vec<String> = table.values().map(|code| format!("{code:?}")).collect();
    let mut word_to_code = phf_codegen::Map::new();
    for ((word, _code), code_literal) in table.iter().zip(&code_literals) {
        word_to_code.entry(word.as_str(), code_literal);
    }
    writeln!(
        out,
        "pub static WORD_TO_CODE: phf::Map<&'static str, &'static str> = {};",
        word_to_code.build()
    )
    .unwrap();

    let word_literals: Vec<String> = table.keys().map(|word| format!("{word:?}")).collect();
    let mut code_to_word = phf_codegen::Map::new();
    for ((_word, code), word_literal) in table.iter().zip(&word_literals) {
        code_to_word.entry(code.as_str(), word_literal);
    }
    writeln!(
        out,
        "pub static CODE_TO_WORD: phf::Map<&'static str, &'static str> = {};",
        code_to_word.build()
    )
    .unwrap();
}

/// Minimal parser for the flat `{"word": "CODE", ...}` shape of
/// `dict_en.json` -- avoids pulling in a JSON crate just for a build-time,
/// one-shot, known-simple-shape file. Values in this file are always plain
/// ASCII strings with no escapes, so this doesn't need general JSON
/// escaping support.
fn parse_json_string_map(raw: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut chars = raw.char_indices().peekable();
    let bytes = raw.as_bytes();

    let read_string = |start: usize| -> (String, usize) {
        // start points at the opening quote
        let mut i = start + 1;
        let mut s = String::new();
        while bytes[i] != b'"' {
            if bytes[i] == b'\\' {
                i += 1;
                s.push(bytes[i] as char);
            } else {
                s.push(bytes[i] as char);
            }
            i += 1;
        }
        (s, i + 1) // position after closing quote
    };

    while let Some(&(i, c)) = chars.peek() {
        if c == '"' {
            let (key, after_key) = read_string(i);
            // advance `chars` past the key
            while chars.peek().map(|&(p, _)| p < after_key).unwrap_or(false) {
                chars.next();
            }
            // skip to the value's opening quote
            let mut j = after_key;
            while bytes[j] != b'"' {
                j += 1;
            }
            let (value, after_value) = read_string(j);
            while chars.peek().map(|&(p, _)| p < after_value).unwrap_or(false) {
                chars.next();
            }
            map.insert(key, value);
        } else {
            chars.next();
        }
    }
    map
}
