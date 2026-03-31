#[cfg(test)]
mod tests {
    use dress_graph::{DRESS, DeltaDressResult, Variant, delta_fit, fit};

    fn hist_total(r: &DeltaDressResult) -> i64 {
        r.histogram.iter().map(|e| e.count).sum()
    }

    fn hist_count_value(r: &DeltaDressResult, value: f64) -> i64 {
        r.histogram.iter()
            .find(|e| (e.value - value).abs() < 1e-9)
            .map(|e| e.count)
            .unwrap_or(0)
    }

    const EPS: f64 = 1e-3;

    // -- One-shot fit tests --

    #[test]
    fn test_triangle() {
        let r = fit(
            3, vec![0, 1, 0], vec![1, 2, 2], None, None,
            Variant::Undirected, false, 100, 1e-8,
        ).unwrap();
        assert_eq!(r.sources.len(), 3);
        assert!(r.iterations > 0);
        let d0 = r.edge_dress[0];
        for &d in &r.edge_dress {
            assert!((d - d0).abs() < 1e-6, "triangle edges should be equal");
        }
    }

    #[test]
    fn test_path() {
        let r = fit(
            4, vec![0, 1, 2], vec![1, 2, 3], None, None,
            Variant::Undirected, false, 100, 1e-6,
        ).unwrap();
        assert_eq!(r.edge_dress.len(), 3);
        for &d in &r.edge_dress {
            assert!(d > 0.0);
            assert!(d < 2.0);
        }
        assert!((r.edge_dress[0] - r.edge_dress[2]).abs() < 1e-10);
    }

    #[test]
    fn test_variants() {
        for &v in &[Variant::Undirected, Variant::Directed, Variant::Forward, Variant::Backward] {
            let r = fit(3, vec![0,1,0], vec![1,2,2], None, None, v, false, 100, 1e-6).unwrap();
            assert_eq!(r.sources.len(), 3);
        }
    }

    #[test]
    fn test_weighted() {
        let r = fit(
            3, vec![0,1,0], vec![1,2,2], Some(vec![1.0,2.0,3.0]), None,
            Variant::Undirected, false, 100, 1e-6,
        ).unwrap();
        assert_eq!(r.edge_weight.len(), 3);
    }

    // -- One-shot delta_fit tests --

    #[test]
    fn test_delta_hist_size() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            0, 100, 1e-3, 0, 0, false, true,
        ).unwrap();
        assert_eq!(r.histogram.len(), 1);
        assert_eq!(hist_count_value(&r, 2.0), 3);
    }

    #[test]
    fn test_delta_weighted_hist_size() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], Some(vec![1.0,10.0,1.0]), None,
            Variant::Undirected, false,
            0, 100, 1e-3, 0, 0, false, true,
        ).unwrap();
        assert!(r.histogram.len() > 1, "weighted should have multiple entries");
        assert_eq!(hist_total(&r), 3);
    }

    #[test]
    fn test_delta0_k3() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 3);
        assert_eq!(r.histogram.len(), 1);
        assert_eq!(hist_count_value(&r, 2.0), 3);
    }

    #[test]
    fn test_delta1_k3() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            1, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 3);
    }

    #[test]
    fn test_delta2_k3() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            2, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 0);
    }

    #[test]
    fn test_delta0_k4() {
        let r = delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 6);
        assert_eq!(hist_count_value(&r, 2.0), 6);
    }

    #[test]
    fn test_delta1_k4() {
        let r = delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None, None,
            Variant::Undirected, false,
            1, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 12);
        assert_eq!(hist_count_value(&r, 2.0), 12);
    }

    #[test]
    fn test_delta2_k4() {
        let r = delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None, None,
            Variant::Undirected, false,
            2, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 6);
    }

    #[test]
    fn test_delta_k_geq_n() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            3, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 0);
    }

    #[test]
    fn test_delta_precompute() {
        let r1 = delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None, None,
            Variant::Undirected, false,
            1, 100, EPS, 0, 0, false, true,
        ).unwrap();
        let r2 = delta_fit(
            4, vec![0,0,0,1,1,2], vec![1,2,3,2,3,3], None, None,
            Variant::Undirected, true,
            1, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(r1.histogram, r2.histogram);
    }

    #[test]
    fn test_delta_path_p4() {
        let r = delta_fit(
            4, vec![0,1,2], vec![1,2,3], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 3);
        assert!(r.histogram.len() >= 2, "path P4 should have >= 2 distinct values");
    }

    #[test]
    fn test_delta1_p4() {
        let r = delta_fit(
            4, vec![0,1,2], vec![1,2,3], None, None,
            Variant::Undirected, false,
            1, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert_eq!(hist_total(&r), 6);
    }

    #[test]
    fn test_delta_length_mismatch() {
        let r = delta_fit(
            3, vec![0, 1], vec![1, 2, 2], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, false, true,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_delta_display() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, false, true,
        ).unwrap();
        let s = format!("{}", r);
        assert!(s.contains("DeltaDressResult"));
        assert!(s.contains("histogram_entries="));
    }

    // -- multisets --

    #[test]
    fn test_multisets_disabled() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, false, true,
        ).unwrap();
        assert!(r.multisets.is_none());
    }

    #[test]
    fn test_multisets_delta0_k3() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            0, 100, EPS, 0, 0, true, true,
        ).unwrap();
        assert_eq!(r.num_subgraphs, 1);
        let ms = r.multisets.unwrap();
        assert_eq!(ms.len(), 3);
        for (i, &v) in ms.iter().enumerate() {
            assert!((v - 2.0).abs() < EPS, "ms[{}] = {}, expected ~2.0", i, v);
        }
    }

    #[test]
    fn test_multisets_delta1_k3() {
        let r = delta_fit(
            3, vec![0,1,0], vec![1,2,2], None, None,
            Variant::Undirected, false,
            1, 100, EPS, 0, 0, true, true,
        ).unwrap();
        assert_eq!(r.num_subgraphs, 3);
        let ms = r.multisets.unwrap();
        assert_eq!(ms.len(), 9);
        let e = 3;
        for s in 0..3 {
            let nans = (0..e).filter(|&j| ms[s * e + j].is_nan()).count();
            assert_eq!(nans, 2, "row {}: expected 2 NaN", s);
            let vals: Vec<f64> = (0..e).map(|j| ms[s*e+j]).filter(|v| !v.is_nan()).collect();
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 2.0).abs() < EPS, "row {}: val={}", s, vals[0]);
        }
    }

    // -- Persistent DRESS object tests --

    #[test]
    fn test_dress_graph_triangle() {
        let mut g = DRESS::new(
            3, vec![0, 1, 0], vec![1, 2, 2], None, None,
            Variant::Undirected, false,
        ).unwrap();

        let (iters, delta) = g.fit(100, 1e-8);
        assert!(iters >= 1);
        assert!(delta < 1e-6);

        let r = g.result();
        let d0 = r.edge_dress[0];
        for &d in &r.edge_dress {
            assert!((d - d0).abs() < 1e-6);
        }

        let d01 = g.get(0, 1, 100, 1e-8, 1.0);
        assert!((d01 - d0).abs() < 1e-4);

        let _ = g.get(0, 0, 100, 1e-6, 1.0);

        let dr = g.delta_fit(1, 100, EPS, 0, 0, false, true).unwrap();
        assert_eq!(hist_total(&dr), 3);

        g.close();
    }

    #[test]
    fn test_dress_graph_drop() {
        let mut g = DRESS::new(
            3, vec![0, 1, 0], vec![1, 2, 2], None, None,
            Variant::Undirected, false,
        ).unwrap();
        g.fit(10, 1e-6);
    }
}
