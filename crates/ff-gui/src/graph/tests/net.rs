//! Tests for module `net`: nodes, links along the data path, per-layer feeds, and the landmarks
//! the summary names.

use super::*;
use crate::graph::tests::{checkpoint, multimodal};
use crate::graph::towers;

fn net_of(i: &Inspected) -> Net {
    build(i, &towers(i))
}

fn find(net: &Net, pred: impl Fn(&What) -> bool) -> usize {
    net.nodes.iter().position(|n| pred(&n.what)).expect("node present")
}

#[test]
fn nodes_and_links_follow_the_data_path() {
    let i = multimodal();
    let net = net_of(&i);
    let words = find(&net, |w| matches!(w, What::Words(_)));
    let per_layer = find(&net, |w| matches!(w, What::PerLayer { .. }));
    let bridge = find(&net, |w| matches!(w, What::Bridge(_)));
    let output = find(&net, |w| matches!(w, What::Output(_)));
    let block = |stack: &str, index: usize| {
        find(
            &net,
            |w| matches!(w, What::Block { stack: s, index: x, .. } if s.contains(stack) && *x == index),
        )
    };
    let (l0, l1, l2, v0) =
        (block("language", 0), block("language", 1), block("language", 2), block("vision", 0));
    let has = |a, b, link| net.links.contains(&(a, b, link));
    assert!(has(words, l0, Link::Flow), "words enter the first block");
    assert!(has(l0, l1, Link::Flow) && has(l1, l2, Link::Flow));
    assert!(has(l2, output, Link::Flow), "the last block feeds the final norm");
    assert!(has(v0, bridge, Link::Flow) && has(bridge, l0, Link::Flow), "vision joins the language part");
    assert!(
        [l0, l1, l2].iter().all(|&b| has(per_layer, b, Link::Feed)),
        "the per-layer table feeds every block"
    );
    assert_eq!(net.nodes[per_layer].params, 100 * 12);
    assert!(
        matches!(&net.nodes[l2].what, What::Block { shared_memory: true, kind: Some(k), .. } if k == "full_attention")
    );
    assert!(matches!(&net.nodes[l0].what, What::Block { shared_memory: false, .. }));
}

#[test]
fn landmarks_name_the_largest_entry_exit_and_hub() {
    let i = multimodal();
    let net = net_of(&i);
    let m = landmarks(&net);
    assert!(matches!(net.nodes[m.largest.unwrap()].what, What::PerLayer { .. }), "the 1200-parameter table");
    assert!(matches!(&net.nodes[m.entry.unwrap()].what, What::Words(n) if n.role == Role::TokenEmbedding));
    assert!(matches!(net.nodes[m.exit.unwrap()].what, What::Output(_)));
    // The first language block takes the words, the bridge, the per-layer feed and the next block.
    assert!(
        matches!(&net.nodes[m.most_linked.unwrap()].what, What::Block { index: 0, stack, .. } if stack.contains("language"))
    );
}

#[test]
fn a_plain_chain_has_no_hub_and_unknown_models_still_get_nodes() {
    let i = checkpoint("chain", &[("net.blocks.0.w", &[4, 4]), ("net.blocks.1.w", &[4, 4])], None);
    let net = net_of(&i);
    assert_eq!(net.nodes.len(), 2);
    assert_eq!(net.links, [(0, 1, Link::Flow)]);
    let m = landmarks(&net);
    assert_eq!((m.entry, m.exit, m.most_linked), (None, None, None));
    assert!(m.largest.is_some());
}
