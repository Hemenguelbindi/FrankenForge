//! Tests for module `vocab`: byte-level decoding, word lookup, pieces of a split word, loading.

use std::fs;

use super::*;

#[test]
fn byte_level_tokens_decode_to_text() {
    assert_eq!(from_byte_level("Ġcat"), " cat");
    assert_eq!(from_byte_level("Ċ"), "\n");
    // "кот" as GPT-2 byte stand-ins (UTF-8 d0 ba d0 be d1 82).
    assert_eq!(from_byte_level("ÐºÐ¾ÑĤ"), "кот");
    assert_eq!(byte_of('A'), Some(b'A'));
    assert_eq!(byte_of('Ā'), Some(0), "U+0100 stands for byte 0");
    assert_eq!(byte_of('中'), None);
    assert_eq!(from_byte_level("中"), "中", "not byte-level: kept as is");
}

#[test]
fn find_prefers_the_word_start_form() {
    let v = Vocab::from_texts(vec!["cat".into(), " cat".into(), "dog".into(), String::new()]);
    assert_eq!(v.find("cat"), Some(1));
    assert_eq!(v.find(" dog "), Some(2));
    assert_eq!(v.find("bird"), None);
    assert_eq!(v.text(3), "<3>", "empty slot");
    assert_eq!(v.text(99), "<99>", "padding row past the vocabulary");
}

#[test]
fn pieces_of_a_split_word_longest_first() {
    let v = Vocab::from_texts(vec![" ко".into(), "к".into(), " кот".into(), "т".into(), " собака".into()]);
    let pieces: Vec<String> = v.pieces_of("Котик", 5).into_iter().map(|(_, t)| t).collect();
    assert_eq!(pieces, [" кот", " ко"]);
}

#[test]
fn loads_byte_level_and_sentencepiece_tokenizers() {
    let dir = std::env::temp_dir().join(format!("ff-lens-vocab-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let byte_level = r#"{"model": {"vocab": {"Ġcat": 0, "ÐºÐ¾ÑĤ": 2}}, "decoder": {"type": "ByteLevel"},
                        "added_tokens": [{"id": 3, "content": "<eos>"}]}"#;
    fs::write(dir.join("tokenizer.json"), byte_level).unwrap();
    let v = Vocab::load(&dir).unwrap().unwrap();
    assert_eq!(
        (v.text(0), v.text(1), v.text(2), v.text(3)),
        (" cat".into(), "<1>".into(), "кот".into(), "<eos>".into())
    );

    let sentencepiece = r#"{"model": {"vocab": [["▁кот", 0.0], ["猫", 0.0]]}, "decoder": {"type": "Sequence", "decoders": []}}"#;
    fs::write(dir.join("tokenizer.json"), sentencepiece).unwrap();
    let v = Vocab::load(&dir).unwrap().unwrap();
    assert_eq!(v.find("кот"), Some(0));
    assert_eq!(v.text(1), "猫");

    fs::write(dir.join("tokenizer.json"), r#"{"model": {"vocab": {"x": 99999999}}}"#).unwrap();
    assert!(Vocab::load(&dir).is_err(), "an absurd id is a damaged file, not a huge allocation");
    fs::remove_dir_all(&dir).unwrap();
    assert_eq!(Vocab::load(&dir).unwrap(), None);
}
