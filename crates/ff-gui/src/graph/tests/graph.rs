//! Tests for module `graph`: towers and blocks from names, shapes and config only, for a known
//! multimodal layout, a GPT-2 style transposed layout and an unrecognised one.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ff_core::{DType, Model, ModelFormat, TensorInfo};
use ff_io::Safetensors;

use super::*;

pub(crate) fn checkpoint(name: &str, tensors: &[(&str, &[usize])], config: Option<&str>) -> Inspected {
    // Tests run in parallel and may build the same fixture: every call gets its own folder.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir: PathBuf = std::env::temp_dir().join(format!("ff-gui-graph-{name}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut m = Model::default();
    for (n, shape) in tensors {
        m.insert(TensorInfo { name: (*n).into(), dtype: DType::BF16, shape: shape.to_vec() });
    }
    Safetensors.write(&dir.join("model.safetensors"), &m, &mut |t| Ok(vec![0; t.byte_len()])).unwrap();
    if let Some(c) = config {
        fs::write(dir.join("config.json"), c).unwrap();
    }
    let inspected = Inspected::open(&dir, &Safetensors).unwrap();
    fs::remove_dir_all(&dir).unwrap();
    inspected
}

/// Gemma-like: three language blocks, a vision tower with a bridge, a per-layer table.
pub(crate) fn multimodal() -> Inspected {
    let mut t: Vec<(String, Vec<usize>)> = Vec::new();
    for i in 0..3 {
        let p = format!("model.language_model.layers.{i}.");
        for (s, shape) in [
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("self_attn.q_norm.weight", vec![2]),
            ("self_attn.q_proj.weight", vec![16, 8]),
            ("self_attn.o_proj.weight", vec![8, 16]),
            ("mlp.gate_proj.weight", vec![32, 8]),
            ("mlp.down_proj.weight", vec![8, 32]),
            ("per_layer_projection.weight", vec![8, 4]),
            ("layer_scalar", vec![1]),
        ] {
            t.push((format!("{p}{s}"), shape));
        }
    }
    t.push(("model.language_model.embed_tokens.weight".into(), vec![100, 8]));
    t.push(("model.language_model.embed_tokens_per_layer.weight".into(), vec![100, 12]));
    t.push(("model.language_model.norm.weight".into(), vec![8]));
    t.push(("model.vision_tower.encoder.layers.0.input_layernorm.weight".into(), vec![6]));
    t.push(("model.vision_tower.encoder.layers.0.mlp.up_proj.weight".into(), vec![12, 6]));
    t.push(("model.embed_vision.embedding_projection.weight".into(), vec![8, 6]));
    let refs: Vec<(&str, &[usize])> = t.iter().map(|(n, s)| (n.as_str(), s.as_slice())).collect();
    let config = r#"{"text_config": {"layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
                     "num_kv_shared_layers": 1}}"#;
    checkpoint("mm", &refs, Some(config))
}

#[test]
fn towers_put_language_last_with_inputs_bridges_and_config_links() {
    let i = multimodal();
    let towers = towers(&i);
    assert_eq!(towers.iter().map(|t| t.kind).collect::<Vec<_>>(), [Kind::Vision, Kind::Language]);
    let (vision, language) = (&towers[0], &towers[1]);
    assert_eq!(vision.width, Some(6));
    assert_eq!(vision.bridges.len(), 1);
    assert_eq!(language.width, Some(8), "from the block norms, not the 2-wide q norm");
    assert_eq!(language.inputs[0].role, Role::TokenEmbedding);
    assert_eq!(language.per_layer[0].name, "model.language_model.embed_tokens_per_layer.weight");
    assert_eq!(language.outputs.len(), 1);
    let s = &language.stacks[0];
    assert_eq!(s.layer_types, ["sliding_attention", "sliding_attention", "full_attention"]);
    assert_eq!(s.kv_shared, Some(1));
}

#[test]
fn block_shows_branches_with_directions_from_roles() {
    let i = multimodal();
    let g = block(&i, 0, Some(8));
    assert_eq!(g.norms, 3);
    let groups: Vec<Group> = g.branches.iter().map(|b| b.group).collect();
    assert_eq!(groups, [Group::Attention, Group::Mlp, Group::Other]);
    let attention = &g.branches[0].matrices;
    assert_eq!(attention[0].flow, Flow::Reads { from: 8, to: 16 }, "q reads the stream");
    assert_eq!(attention[1].flow, Flow::Writes { from: 16, to: 8 }, "o writes back");
    assert_eq!(g.branches[1].matrices[1].flow, Flow::Writes { from: 32, to: 8 });
    let other = &g.branches[2];
    assert_eq!(other.matrices[0].flow, Flow::Writes { from: 4, to: 8 }, "unknown [w, x] writes back");
    assert_eq!(other.small, 1, "layer_scalar is a small part, not a matrix");
}

#[test]
fn transposed_storage_reads_the_same() {
    // GPT-2 keeps Conv1D weights as [in, out]; direction must come from the role.
    assert_eq!(flow(Role::AttnQkv, &[768, 2304], Some(768)), Flow::Reads { from: 768, to: 2304 });
    assert_eq!(flow(Role::Mlp(Part::Down), &[3072, 768], Some(768)), Flow::Writes { from: 3072, to: 768 });
    assert_eq!(flow(Role::AttnQ, &[5, 7], Some(768)), Flow::Shape(vec![5, 7]));
    assert_eq!(flow(Role::AttnQ, &[5, 7], None), Flow::Shape(vec![5, 7]));
}

#[test]
fn unknown_architecture_still_draws_with_real_sizes() {
    let i = checkpoint(
        "odd",
        &[
            ("net.blocks.0.mixer.w", &[4, 4]),
            ("net.blocks.0.mixer.b", &[4]),
            ("net.blocks.1.mixer.w", &[4, 4]),
        ],
        Some(r#"{"layer_types": ["a"]}"#),
    );
    let towers = towers(&i);
    assert_eq!(towers.len(), 1);
    assert!(towers[0].inputs.is_empty());
    assert!(
        towers[0].stacks[0].layer_types.is_empty(),
        "a list that does not match the block count is ignored"
    );
    let g = block(&i, 0, towers[0].width);
    assert_eq!(g.branches.len(), 1);
    assert_eq!(g.branches[0].group, Group::Other);
    assert_eq!(g.branches[0].matrices[0].flow, Flow::Shape(vec![4, 4]), "no width known: shown as stored");
}

#[test]
fn experts_collapse_into_one_row_per_role() {
    let mut t: Vec<(String, Vec<usize>)> = vec![("m.layers.0.input_layernorm.weight".into(), vec![8])];
    for e in 0..4 {
        t.push((format!("m.layers.0.mlp.experts.{e}.down_proj.weight"), vec![8, 16]));
    }
    t.push(("m.embed_tokens.weight".into(), vec![10, 8]));
    let refs: Vec<(&str, &[usize])> = t.iter().map(|(n, s)| (n.as_str(), s.as_slice())).collect();
    let i = checkpoint("moe", &refs, None);
    let g = block(&i, 0, Some(8));
    let experts = g.branches.iter().find(|b| b.group == Group::Experts).expect("experts branch");
    assert_eq!(experts.matrices.len(), 1);
    assert_eq!(experts.matrices[0].count, 4);
}
