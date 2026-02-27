#[cfg(test)]
mod tests {
    use dress_graph::{DRESS, Variant};

    #[test]
    fn test_triangle() {
        // Triangle: 0-1, 1-2, 0-2
        let r = DRESS::builder(3, vec![0, 1, 0], vec![1, 2, 2])
            .variant(Variant::Undirected)
            .max_iterations(100)
            .epsilon(1e-8)
            .build_and_fit()
            .unwrap();

        assert_eq!(r.sources.len(), 3);
        assert!(r.iterations > 0);
        // In a triangle all edges should have the same dress value
        let d0 = r.edge_dress[0];
        for &d in &r.edge_dress {
            assert!((d - d0).abs() < 1e-6, "edge dress values should be equal in a triangle");
        }
    }

    #[test]
    fn test_path() {
        // Path: 0-1-2-3 (no triangles — dress values are positive but lower
        // than in a triangle, because the self-loop constant contributes to
        // the numerator even when there are no common neighbours).
        let r = DRESS::builder(4, vec![0, 1, 2], vec![1, 2, 3])
            .build_and_fit()
            .unwrap();

        assert_eq!(r.edge_dress.len(), 3);
        for &d in &r.edge_dress {
            assert!(d > 0.0, "path dress should be positive (self-loop term)");
            assert!(d < 2.0, "path dress should be well below 2");
        }
        // Endpoint edges (0-1 and 2-3) should be symmetric
        assert!(
            (r.edge_dress[0] - r.edge_dress[2]).abs() < 1e-10,
            "symmetric path edges should have equal dress",
        );
    }

    #[test]
    fn test_variants() {
        for &v in &[Variant::Undirected, Variant::Directed, Variant::Forward, Variant::Backward] {
            let r = DRESS::builder(3, vec![0, 1, 0], vec![1, 2, 2])
                .variant(v)
                .build_and_fit()
                .unwrap();
            assert_eq!(r.sources.len(), 3);
        }
    }

    #[test]
    fn test_weighted() {
        let r = DRESS::builder(3, vec![0, 1, 0], vec![1, 2, 2])
            .weights(vec![1.0, 2.0, 3.0])
            .build_and_fit()
            .unwrap();
        assert_eq!(r.edge_weight.len(), 3);
    }

    // ── Delta-k-DRESS tests ────────────────────────────────────────

    fn hist_total(r: &dress_graph::DeltaDressResult) -> i64 {
        r.histogram.iter().sum()
    }

    const EPS: f64 = 1e-3;

    #[test]
    fn test_delta_hist_size() {
        let r = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            0, 100, 1e-3, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(r.hist_size, 2001);
        assert_eq!(r.histogram.len(), r.hist_size as usize);

        let r2 = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            0, 100, 1e-6, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(r2.hist_size, 2000001);
    }

    #[test]
    fn test_delta0_k3() {
        let r = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            0, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r), 3);
        // Top bin (dress=2.0 for complete graph)
        assert!(r.histogram[r.hist_size as usize - 1] > 0);
        // Single non-zero bin
        let nonzero = r.histogram.iter().filter(|&&x| x > 0).count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn test_delta1_k3() {
        let r = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            1, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        // C(3,1)=3 subgraphs × 1 edge each = 3
        assert_eq!(hist_total(&r), 3);
    }

    #[test]
    fn test_delta2_k3() {
        let r = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            2, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r), 0);
    }

    #[test]
    fn test_delta0_k4() {
        let r = DRESS::delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None,
            0, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r), 6);
        assert_eq!(r.histogram[r.hist_size as usize - 1], 6);
    }

    #[test]
    fn test_delta1_k4() {
        let r = DRESS::delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None,
            1, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        // C(4,1)=4 × 3 edges = 12
        assert_eq!(hist_total(&r), 12);
        assert_eq!(r.histogram[r.hist_size as usize - 1], 12);
    }

    #[test]
    fn test_delta2_k4() {
        let r = DRESS::delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None,
            2, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        // C(4,2)=6 × 1 edge = 6
        assert_eq!(hist_total(&r), 6);
    }

    #[test]
    fn test_delta_k_geq_n() {
        let r = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            3, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r), 0);

        let r2 = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            10, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r2), 0);
    }

    #[test]
    fn test_delta_precompute() {
        let r1 = DRESS::delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None,
            1, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        let r2 = DRESS::delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None,
            1, 100, EPS, Variant::Undirected, true,
        ).unwrap();
        assert_eq!(r1.hist_size, r2.hist_size);
        assert_eq!(r1.histogram, r2.histogram);
    }

    #[test]
    fn test_delta_path_p4() {
        let r = DRESS::delta_fit(
            4, vec![0,1,2], vec![1,2,3], None,
            0, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r), 3);
        let nonzero = r.histogram.iter().filter(|&&x| x > 0).count();
        assert!(nonzero >= 2, "path P4 should have at least 2 distinct bins");
    }

    #[test]
    fn test_delta1_p4() {
        let r = DRESS::delta_fit(
            4, vec![0,1,2], vec![1,2,3], None,
            1, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        assert_eq!(hist_total(&r), 6);
    }

    #[test]
    fn test_delta_length_mismatch() {
        let r = DRESS::delta_fit(
            3, vec![0, 1], vec![1, 2, 2], None,
            0, 100, EPS, Variant::Undirected, false,
        );
        assert!(r.is_err(), "mismatched lengths should return Err");
    }

    #[test]
    fn test_delta_display() {
        let r = DRESS::delta_fit(
            3, vec![0,1,0], vec![1,2,2], None,
            0, 100, EPS, Variant::Undirected, false,
        ).unwrap();
        let s = format!("{}", r);
        assert!(s.contains("DeltaDressResult"), "Display should contain type name");
    }
}
