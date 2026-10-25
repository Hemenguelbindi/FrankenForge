//! Tests for module `components`.

use super::*;
use ff_core::{DType, TensorInfo};

fn model(names: &[&str]) -> Model {
    let mut m = Model::default();
    for name in names {
        m.insert(TensorInfo { name: (*name).into(), dtype: DType::BF16, shape: vec![2] });
    }
    m
}

fn kinds(cs: &[Component]) -> Vec<Kind> {
    cs.iter().map(|c| c.kind).collect()
}

/// Shape of gemma-4-E2B: audio, language and vision stacks plus loose tensors of each.
#[test]
fn multimodal_names_split_into_components() {
    let m = model(&[
        "model.audio_tower.layers.0.w",
        "model.audio_tower.layers.1.w",
        "model.audio_tower.output_proj.weight",
        "model.embed_audio.embedding_projection.weight",
        "model.language_model.layers.0.w",
        "model.language_model.embed_tokens.weight",
        "model.vision_tower.encoder.layers.0.w",
        "model.embed_vision.embedding_projection.weight",
        "lm_head.weight",
    ]);
    let cs = components(&m, &m.layout());
    assert_eq!(kinds(&cs), [Kind::Language, Kind::Audio, Kind::Vision, Kind::Other]);

    let audio = &cs[1];
    assert_eq!(audio.stacks, [Stack { name: "model.audio_tower.layers".into(), blocks: vec![0, 1] }]);
    assert_eq!(
        audio.loose,
        ["model.audio_tower.output_proj.weight", "model.embed_audio.embedding_projection.weight"]
    );
    assert_eq!(audio.params, 8);
    assert_eq!(cs[0].loose, ["model.language_model.embed_tokens.weight"]);
    assert_eq!(cs[2].stacks[0].name, "model.vision_tower.encoder.layers");
    assert_eq!(cs[3].loose, ["lm_head.weight"]);
    assert_eq!(cs.iter().map(|c| c.params).sum::<usize>(), m.param_count());
}

/// A text-only model gets no audio or vision card, and its loose tensors go to Language.
#[test]
fn text_only_model_claims_no_modality() {
    let m =
        model(&["model.layers.0.w", "model.layers.1.w", "model.embed_tokens.weight", "model.norm.weight"]);
    let cs = components(&m, &m.layout());
    assert_eq!(kinds(&cs), [Kind::Language]);
    assert_eq!(cs[0].block_count(), 2);
    assert_eq!(cs[0].loose.len(), 2);
}

/// Without language words or a token table a stack is not called language.
#[test]
fn stack_without_evidence_is_other() {
    // Whisper-like: only the decoder sits next to `embed_tokens`.
    let m = model(&[
        "model.encoder.layers.0.w",
        "model.encoder.embed_positions.weight",
        "model.decoder.layers.0.w",
        "model.decoder.embed_tokens.weight",
        "mtp.layers.0.w",
    ]);
    let cs = components(&m, &m.layout());
    assert_eq!(kinds(&cs), [Kind::Language, Kind::Other]);
    assert_eq!(cs[0].stacks.len(), 1);
    assert_eq!(cs[0].stacks[0].name, "model.decoder.layers");
    assert_eq!(cs[0].loose, ["model.decoder.embed_tokens.weight"]);
    let other: Vec<&str> = cs[1].stacks.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(other, ["model.encoder.layers", "mtp.layers"]);
    assert_eq!(cs[1].loose, ["model.encoder.embed_positions.weight"]);
}

/// `$env:FF_REAL_MODEL='F:\models\gemma-4-E2B-it\model.safetensors'; cargo test -p ff-gui real -- --ignored --nocapture`
#[test]
#[ignore = "needs FF_REAL_MODEL"]
fn real_model_components() {
    use ff_core::ModelFormat;
    let path = std::path::PathBuf::from(std::env::var_os("FF_REAL_MODEL").expect("set FF_REAL_MODEL"));
    let m = ff_io::Safetensors.open(&path).unwrap().model().clone();
    let cs = components(&m, &m.layout());
    for c in &cs {
        let stacks: Vec<String> =
            c.stacks.iter().map(|s| format!("{} ({})", s.name, s.blocks.len())).collect();
        println!(
            "{:?}: {} blocks, {} loose, {} params; {stacks:?}",
            c.kind,
            c.block_count(),
            c.loose.len(),
            c.params
        );
    }
    assert_eq!(cs.iter().map(|c| c.params).sum::<usize>(), m.param_count());
}

#[test]
fn modality_needs_an_explicit_name_segment() {
    assert_eq!(modality("model.visual.blocks.0.w"), Some(Kind::Vision));
    assert_eq!(modality("model.audio_tower.x"), Some(Kind::Audio));
    // Similar-looking words are not evidence.
    assert_eq!(modality("model.layers.0.self_attn.v_proj.weight"), None);
    assert_eq!(modality("model.revision.w"), None);
}
