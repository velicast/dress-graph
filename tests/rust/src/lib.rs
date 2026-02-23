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
}
