use faer::{Col, Mat};

use mcqgam::model::Regressor;
use mcqgam::solver::objective::{huber_pinball_grad, huber_pinball_hess, huber_pinball_loss};
use mcqgam::solver::pava::pava;

#[test]
fn test_huber_pinball_properties() {
    let quantile = 0.5;
    let epsilon = 0.1;
    let r_values = vec![-0.6, -0.051, -0.049, 0.0, 0.049, 0.051, 0.6];

    for &r in &r_values {
        let h = 1e-6;
        let grad_numerical = (huber_pinball_loss(r + h, quantile, epsilon)
            - huber_pinball_loss(r - h, quantile, epsilon))
            / (2.0 * h);
        let grad_analytical = huber_pinball_grad(r, quantile, epsilon);

        assert!(
            (grad_numerical - grad_analytical).abs() < 1e-9,
            "Grad mismatch at r={}: expected {}, got {}",
            r,
            grad_analytical,
            grad_numerical
        );

        let h2 = 1e-5;
        let hess_numerical = (huber_pinball_grad(r + h2, quantile, epsilon)
            - huber_pinball_grad(r - h2, quantile, epsilon))
            / (2.0 * h2);
        let hess_analytical = huber_pinball_hess(r, quantile, epsilon);

        assert!(
            (hess_numerical - hess_analytical).abs() < 1e-9,
            "Hess mismatch at r={}: expected {}, got {}",
            r,
            hess_analytical,
            hess_numerical
        );
    }
}

#[test]
fn test_pava_basic() {
    let y = vec![1.0, 3.0, 2.0, 4.0];
    let mut z = vec![0.0; 4];
    pava(&y, &mut z);
    assert_eq!(z, vec![1.0, 2.5, 2.5, 4.0]);
}

#[test]
fn test_pava_decreasing() {
    let y = vec![4.0, 3.0, 2.0, 1.0];
    let mut z = vec![0.0; 4];
    pava(&y, &mut z);
    assert_eq!(z, vec![2.5, 2.5, 2.5, 2.5]);
}

#[test]
fn test_pava_increasing() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let mut z = vec![0.0; 4];
    pava(&y, &mut z);
    assert_eq!(z, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_solve_admm_simple() {
    let n_samples = 21;
    let y = Col::<f64>::from_fn(n_samples, |r_idx| r_idx as f64);
    let x = Mat::<f64>::from_fn(n_samples, 1, |r_idx, _| {
        r_idx as f64 / (n_samples - 1) as f64
    });

    let mut model = Regressor::new(1, 0, vec![0.5]);
    model.fit(x.as_ref(), y.as_ref());

    let _coef = model.coeffs.as_ref().unwrap();

    let pred = model.predict(x.as_ref(), vec![0.5]);
    let mid_val = pred[(n_samples / 2, 0)];
    assert!((mid_val - 10.0).abs() < 1.0);
}
