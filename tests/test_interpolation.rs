use mcqgam::interpolation::cubic::MonotoneCubicSpline;

#[test]
fn test_monotone_cubic_spline_exact_points() {
    let quantiles = vec![0.1, 0.5, 0.9];
    let preds = vec![10.0, 20.0, 30.0];
    let spline = MonotoneCubicSpline::new(quantiles.clone(), preds.clone());

    for (q, p) in quantiles.iter().zip(preds.iter()) {
        assert!((spline.interpolate(*q) - *p).abs() < 1e-10);
    }
}

#[test]
fn test_monotone_cubic_spline_monotonicity() {
    let quantiles = vec![0.0, 0.5, 1.0];
    let preds = vec![0.0, 50.0, 100.0];
    let spline = MonotoneCubicSpline::new(quantiles, preds);

    let mut prev_val = -1.0;
    for q_idx in 0..=100 {
        let q = q_idx as f64 / 100.0;
        let val = spline.interpolate(q);
        assert!(val >= prev_val, "Spline lost monotonicity at q={}", q);
        prev_val = val;
    }
}

#[test]
fn test_monotone_cubic_spline_extrapolation() {
    let quantiles = vec![0.2, 0.8];
    let preds = vec![10.0, 20.0];
    let spline = MonotoneCubicSpline::new(quantiles, preds);
    let val_below = spline.interpolate(0.0);
    let expected_below = 10.0 + 16.6666666667 * (0.0 - 0.2);
    assert!((val_below - expected_below).abs() < 1e-10);
}

#[test]
fn test_monotone_cubic_spline_edge_cases() {
    let spline1 = MonotoneCubicSpline::new(vec![0.5], vec![100.0]);
    assert_eq!(spline1.interpolate(0.2), 100.0);
    assert_eq!(spline1.interpolate(0.8), 100.0);

    let spline0 = MonotoneCubicSpline::new(vec![], vec![]);
    assert_eq!(spline0.interpolate(0.5), 0.0);
}
