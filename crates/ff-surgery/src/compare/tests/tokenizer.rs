//! Tests for module `tokenizer`.

use super::*;

const BPE: &str = r#"{"model":{"vocab":{"a":0,"b":1}},"added_tokens":[{"id":2,"content":"<s>"}]}"#;

fn dir_with(name: &str, tokenizer: Option<&str>) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ff-tok-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    if let Some(t) = tokenizer {
        fs::write(d.join(FILE), t).unwrap();
    }
    d
}

#[test]
fn classifies_tokenizers() {
    let same = dir_with("same", Some(BPE));
    let reformatted = dir_with("reformatted", Some(&BPE.replace(',', ", ")));
    let other = dir_with("other", Some(r#"{"model":{"vocab":{"a":0,"c":1}}}"#));
    let none = dir_with("none", None);
    let status = |x: &Path, y: &Path| compare(x, y).unwrap().status;

    assert_eq!(status(&same, &same), TokenizerMatch::Identical);
    assert_eq!(status(&same, &reformatted), TokenizerMatch::SameVocab);
    assert_eq!(status(&same, &other), TokenizerMatch::DifferentVocab);
    assert_eq!(status(&same, &none), TokenizerMatch::Missing);
    let d = compare(&same, &none).unwrap();
    assert_eq!((d.vocab_a, d.vocab_b), (Some(3), None));

    for d in [same, reformatted, other, none] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn unigram_vocab_uses_positions() {
    let v = vocab(Path::new("t"), br#"{"model":{"vocab":[["a",0.0],["b",-1.5]]}}"#).unwrap();
    assert_eq!(v["b"], 1);
}

#[test]
fn rejects_malformed_vocab() {
    assert!(vocab(Path::new("t"), br#"{"model":{}}"#).is_err());
    assert!(vocab(Path::new("t"), br#"{"model":{"vocab":{"a":"x"}}}"#).is_err());
    assert!(vocab(Path::new("t"), b"not json").is_err());
}
