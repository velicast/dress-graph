// mpi.rs — Rook vs Shrikhande with Δ¹-DRESS (MPI, CPU backend)
// Keeps multisets and compares them to guarantee distinguishability.
//
// Add to Cargo.toml:
//   [dependencies]
//   dress-graph = { path = "../../rust", features = ["mpi"] }
//   mpi = "0.8"
//
// Run:
//   mpirun -np 4 cargo run --example mpi --features mpi
use dress_graph::{mpi, Variant};

fn main() {
    let universe = mpi::mpi_crate::initialize().unwrap();
    let world = universe.world();
    let rank = mpi::mpi_crate::topology::Communicator::rank(&world);

    // Rook L₂(4) = K₄ □ K₄ — 16 vertices, 96 directed edges
    let rook_s = vec![0,1,0,4,0,2,0,8,0,3,0,12,1,5,1,2,1,9,1,3,1,13,2,6,2,10,2,3,2,14,3,7,3,11,3,15,4,5,4,6,4,8,4,7,4,12,5,6,5,9,5,7,5,13,6,10,6,7,6,14,7,11,7,15,8,9,8,10,8,11,8,12,9,10,9,11,9,13,10,11,10,14,11,15,12,13,12,14,12,15,13,14,13,15,14,15];
    let rook_t = vec![1,0,4,0,2,0,8,0,3,0,12,0,5,1,2,1,9,1,3,1,13,1,6,2,10,2,3,2,14,2,7,3,11,3,15,3,5,4,6,4,8,4,7,4,12,4,6,5,9,5,7,5,13,5,10,6,7,6,14,6,11,7,15,7,9,8,10,8,11,8,12,8,10,9,11,9,13,9,11,10,14,10,15,11,13,12,14,12,15,12,14,13,15,13,15,14];

    // Shrikhande — 16 vertices, 96 directed edges
    let shri_s = vec![0,4,0,12,0,1,0,3,0,5,0,15,1,5,1,13,1,2,1,6,1,12,2,6,2,14,2,3,2,7,2,13,3,7,3,15,3,4,3,14,4,8,4,5,4,7,4,9,5,9,5,6,5,10,6,10,6,7,6,11,7,11,7,8,8,12,8,9,8,11,8,13,9,13,9,10,9,14,10,14,10,11,10,15,11,15,11,12,12,13,12,15,13,14,14,15];
    let shri_t = vec![4,0,12,0,1,0,3,0,5,0,15,0,5,1,13,1,2,1,6,1,12,1,6,2,14,2,3,2,7,2,13,2,7,3,15,3,4,3,14,3,8,4,5,4,7,4,9,4,9,5,6,5,10,5,10,6,7,6,11,6,11,7,8,7,12,8,9,8,11,8,13,8,13,9,10,9,14,9,14,10,11,10,15,10,15,11,12,11,13,12,15,12,14,13,15,14];

    let dr = mpi::delta_fit(16, rook_s, rook_t, None, None, 1, 100, 1e-6, 0, 0,
                            Variant::Undirected, false, true, true, &world).unwrap();
    let ds = mpi::delta_fit(16, shri_s, shri_t, None, None, 1, 100, 1e-6, 0, 0,
                            Variant::Undirected, false, true, true, &world).unwrap();

    if rank == 0 {
        println!("Rook:       {} bins, {} subgraphs", dr.histogram.len(), dr.num_subgraphs);
        println!("Shrikhande: {} bins, {} subgraphs", ds.histogram.len(), ds.num_subgraphs);
        println!("Histograms differ:  {}", dr.histogram != ds.histogram);

        // Canonicalize multisets: sort each row, then sort rows
        let canonicalize = |ms: &[f64], ns: usize, e: usize| -> Vec<Vec<f64>> {
            let mut rows: Vec<Vec<f64>> = (0..ns)
                .map(|i| {
                    let mut row = ms[i * e..(i + 1) * e].to_vec();
                    row.sort_by(|a, b| {
                        if a.is_nan() && b.is_nan() { std::cmp::Ordering::Equal }
                        else if a.is_nan() { std::cmp::Ordering::Greater }
                        else if b.is_nan() { std::cmp::Ordering::Less }
                        else { a.partial_cmp(b).unwrap() }
                    });
                    row
                })
                .collect();
            rows.sort_by(|a, b| {
                a.iter().zip(b.iter()).find_map(|(x, y)| {
                    let c = if x.is_nan() && y.is_nan() { std::cmp::Ordering::Equal }
                            else if x.is_nan() { std::cmp::Ordering::Greater }
                            else if y.is_nan() { std::cmp::Ordering::Less }
                            else { x.partial_cmp(y).unwrap() };
                    if c != std::cmp::Ordering::Equal { Some(c) } else { None }
                }).unwrap_or(std::cmp::Ordering::Equal)
            });
            rows
        };

        let e = 96usize;
        let cr = canonicalize(dr.multisets.as_ref().unwrap(), dr.num_subgraphs as usize, e);
        let cs = canonicalize(ds.multisets.as_ref().unwrap(), ds.num_subgraphs as usize, e);

        let ms_same = cr.len() == cs.len() && cr.iter().zip(cs.iter()).all(|(a, b)| {
            a.iter().zip(b.iter()).all(|(x, y)| {
                (x.is_nan() && y.is_nan()) || x == y
            })
        });
        println!("Multisets differ:   {}", !ms_same);
    }
}
