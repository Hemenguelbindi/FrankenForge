//! Tests for module `layout`: placement is deterministic, finite, keeps circles apart and puts
//! linked nodes closer than unlinked ones.

use super::*;
use crate::graph::net::build;
use crate::graph::tests::multimodal;
use crate::graph::towers;

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

#[test]
fn placement_is_stable_spread_and_follows_links() {
    let i = multimodal();
    let net = build(&i, &towers(&i));
    let radius: Vec<f32> = net.nodes.iter().map(|n| 7.0 + (n.params as f32).sqrt() / 4.0).collect();
    let pos = place(&net, &radius);
    assert_eq!(pos, place(&net, &radius), "same model, same picture");
    assert!(pos.iter().flatten().all(|v| v.is_finite()));

    let n = pos.len();
    for a in 0..n {
        for b in a + 1..n {
            assert!(dist(pos[a], pos[b]) > (radius[a] + radius[b]) * 0.8, "circles {a} and {b} overlap");
        }
    }
    let linked: Vec<f32> =
        net.links.iter().filter(|l| l.2 == Link::Flow).map(|l| dist(pos[l.0], pos[l.1])).collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let all: Vec<f32> =
        (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).map(|(a, b)| dist(pos[a], pos[b])).collect();
    assert!(mean(&linked) < mean(&all), "linked {} vs all {}", mean(&linked), mean(&all));
}
