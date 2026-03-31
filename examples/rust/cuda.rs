// cuda.rs — Prism vs K₃,₃ with DRESS (CUDA)
//
// Add to Cargo.toml:
//   [dependencies]
//   dress-graph = { path = "../../rust", features = ["cuda"] }
//
// Run:
//   cargo run --example cuda --features cuda
use dress_graph::cuda::{DRESS, Variant};

fn main() {
    // Prism (C₃ □ K₂): 6 vertices, 18 directed edges
    let prism_s = vec![0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3];
    let prism_t = vec![1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5];

    // K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
    let k33_s = vec![0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5];
    let k33_t = vec![3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2];

    let rp = {
        let mut _g = DRESS::new(6, prism_s, prism_t, None, None, Variant::Undirected, false).unwrap();
        _g.fit(100, 1e-6);
        _g.result()
    };

    let rk = {
        let mut _g = DRESS::new(6, k33_s, k33_t, None, None, Variant::Undirected, false).unwrap();
        _g.fit(100, 1e-6);
        _g.result()
    };

    let mut fp: Vec<f64> = rp.edge_dress.clone();
    let mut fk: Vec<f64> = rk.edge_dress.clone();
    fp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fk.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Prism: {:?}", fp.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>());
    println!("K3,3:  {:?}", fk.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>());
    println!("Distinguished: {}", fp != fk);
}
