//! Free-form placement of the graph, as in note-taking graph views: linked nodes pull together,
//! all nodes push apart, a weak pull keeps the whole near the centre. Deterministic (no
//! randomness), so the same model always looks the same and tests can check it.

use super::net::{Link, Net};

/// Preferred distance between linked nodes, in canvas points.
const SPACING: f32 = 70.0;
const STEPS: usize = 350;

/// Positions for every node; `radius[i]` keeps big nodes from overlapping their neighbours.
pub fn place(net: &Net, radius: &[f32]) -> Vec<[f32; 2]> {
    let n = net.nodes.len();
    // Sunflower start: spread out, and neighbours in the list start near each other.
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let mut pos: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let r = SPACING * 0.6 * (i as f32 + 0.5).sqrt();
            let a = i as f32 * golden;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    let k2 = SPACING * SPACING;
    for step in 0..STEPS {
        let heat = SPACING * (1.0 - step as f32 / STEPS as f32) + 1.0;
        let mut push = vec![[0f32; 2]; n];
        for i in 0..n {
            for j in i + 1..n {
                let (dx, dy) = (pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]);
                let d = (dx * dx + dy * dy).sqrt().max(0.5);
                let room = radius[i] + radius[j] + 12.0;
                let f = k2 / d + if d < room { (room - d) * 4.0 } else { 0.0 };
                let (fx, fy) = (dx / d * f, dy / d * f);
                push[i][0] += fx;
                push[i][1] += fy;
                push[j][0] -= fx;
                push[j][1] -= fy;
            }
        }
        for &(a, b, link) in &net.links {
            let (dx, dy) = (pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]);
            let d = (dx * dx + dy * dy).sqrt().max(0.5);
            // A table feeding every block would otherwise crush them into a ball around it.
            let weight = if link == Link::Feed { 0.08 } else { 1.0 };
            let f = d * d / SPACING * weight;
            let (fx, fy) = (dx / d * f, dy / d * f);
            push[a][0] -= fx;
            push[a][1] -= fy;
            push[b][0] += fx;
            push[b][1] += fy;
        }
        for (p, f) in pos.iter_mut().zip(&push) {
            let (fx, fy) = (f[0] - p[0] * 0.05, f[1] - p[1] * 0.05);
            let len = (fx * fx + fy * fy).sqrt().max(1e-6);
            let step = len.min(heat);
            p[0] += fx / len * step;
            p[1] += fy / len * step;
        }
    }
    pos
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
