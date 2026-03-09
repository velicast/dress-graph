// cpu.rs — Prism vs K₃,₃ with DRESS (CPU)
//
// Add to Cargo.toml:
//   [dependencies]
//   dress-graph = { path = "../../rust" }
//
// Run:
//   cargo run --example cpu
use dress_graph::{DRESS, Variant};

fn main() {
    // Prism (C₃ □ K₂): 6 vertices, 18 directed edges
    let prism_s = vec![0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3];
    let prism_t = vec![1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5];

    // K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
    let k33_s = vec![0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5];
    let k33_t = vec![3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2];

    let rp = DRESS::builder(6, prism_s, prism_t)
        .variant(Variant::Undirected)
        .build_and_fit()
        .unwrap();

    let rk = DRESS::builder(6, k33_s, k33_t)
        .variant(Variant::Undirected)
        .build_and_fit()
        .unwrap();

    let mut fp: Vec<f64> = rp.edge_dress.clone();
    let mut fk: Vec<f64> = rk.edge_dress.clone();
    fp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fk.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Prism: {:?}", fp.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>());
    println!("K3,3:  {:?}", fk.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>());
    println!("Distinguished: {}", fp != fk);
}
