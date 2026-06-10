use faer::Mat;

use mcqgam::bspline::basis::evaluate_bspline;
use mcqgam::bspline::knots::generate_quantile_knots;
use mcqgam::bspline::matrix::StructuredDesignMatrix;

#[test]
fn test_generate_quantile_knots_basic() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let n_internal_knots = 3;
    let degree = 3;
    let knots = generate_quantile_knots(&data, n_internal_knots, degree);

    assert_eq!(knots.len(), 11);
    assert_eq!(knots[0], 0.0);
    assert_eq!(knots[3], 0.0);
    assert_eq!(knots[4], 2.0);
    assert_eq!(knots[5], 5.0);
    assert_eq!(knots[6], 7.0);
    assert_eq!(knots[7], 10.0);
    assert_eq!(knots[10], 10.0);
}

#[test]
fn test_generate_quantile_knots_no_internal() {
    let data = vec![1.0, 2.0, 3.0];
    let knots = generate_quantile_knots(&data, 0, 2);

    assert_eq!(knots.len(), 6);
    assert_eq!(knots, vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_generate_quantile_knots_duplicate_data() {
    let data = vec![1.0, 1.0, 1.0, 1.0];
    let knots = generate_quantile_knots(&data, 2, 1);

    assert_eq!(knots, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_generate_quantile_knots_unsorted() {
    let data = vec![9.0, 1.0, 0.0, 5.0, 2.0, 7.0, 8.0, 4.0, 3.0, 6.0, 10.0];
    let n_internal_knots = 3;
    let degree = 3;
    let knots = generate_quantile_knots(&data, n_internal_knots, degree);

    assert_eq!(knots.len(), 11);
    assert_eq!(knots[0], 0.0);
    assert_eq!(knots[3], 0.0);
    assert_eq!(knots[4], 2.0);
    assert_eq!(knots[5], 5.0);
    assert_eq!(knots[6], 7.0);
    assert_eq!(knots[7], 10.0);
}

#[test]
fn test_generate_quantile_knots_with_nans() {
    let data = vec![f64::NAN, 0.0, f64::NAN, 5.0, 10.0, f64::NAN];
    let knots = generate_quantile_knots(&data, 1, 1);

    assert_eq!(knots, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_generate_quantile_knots_empty_and_all_nan() {
    let empty_data: Vec<f64> = vec![];
    assert!(generate_quantile_knots(&empty_data, 2, 2).is_empty());

    let nan_data = vec![f64::NAN, f64::NAN];
    assert!(generate_quantile_knots(&nan_data, 2, 2).is_empty());
}

#[test]
fn test_generate_quantile_knots_zero_inflated_financial() {
    let data = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 10.0];
    let knots = generate_quantile_knots(&data, 3, 2);

    assert_eq!(knots.len(), 7);
    for i in 1..knots.len() {
        assert!(knots[i] >= knots[i - 1]);
    }
}

#[test]
fn test_partition_of_unity() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    let degree = 3;
    let mut out_basis = vec![0.0; degree + 1];
    let test_points = vec![0.0, 0.25, 0.5, 1.0, 1.5, 1.99, 2.0];

    for &x in &test_points {
        let _start_idx = evaluate_bspline(x, &knots, degree, &mut out_basis);
        for &val in &out_basis {
            assert!(val >= 0.0);
        }
        let sum: f64 = out_basis.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}

#[test]
fn test_exact_values_linear() {
    let knots = vec![0.0, 0.0, 1.0, 2.0, 2.0];
    let degree = 1;
    let mut out_basis = vec![0.0; degree + 1];
    let start_idx = evaluate_bspline(0.5, &knots, degree, &mut out_basis);

    assert_eq!(start_idx, 0);
    assert!((out_basis[0] - 0.5).abs() < 1e-9);
    assert!((out_basis[1] - 0.5).abs() < 1e-9);
}

#[test]
fn test_clamping_and_boundaries() {
    let knots = vec![0.0, 0.0, 1.0, 2.0, 2.0];
    let degree = 1;
    let mut out_basis_under = vec![0.0; degree + 1];
    let mut out_basis_min = vec![0.0; degree + 1];

    evaluate_bspline(-5.0, &knots, degree, &mut out_basis_under);
    evaluate_bspline(0.0, &knots, degree, &mut out_basis_min);

    assert_eq!(out_basis_under, out_basis_min);

    let mut out_basis_over = vec![0.0; degree + 1];
    let mut out_basis_max = vec![0.0; degree + 1];

    evaluate_bspline(10.0, &knots, degree, &mut out_basis_over);
    evaluate_bspline(2.0, &knots, degree, &mut out_basis_max);

    assert_eq!(out_basis_over, out_basis_max);
}

#[test]
fn test_evaluate_bspline_partition_of_unity() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let degree = 3;
    let knots = generate_quantile_knots(&data, 2, degree);

    let mut out_basis = vec![0.0; degree + 1];
    for &x in &[0.0, 0.5, 2.5, 5.0] {
        let _ = evaluate_bspline(x, &knots, degree, &mut out_basis);
        let sum: f64 = out_basis.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}

fn setup_test_matrix(n_samples: usize, n_features: usize, degree: usize) -> StructuredDesignMatrix {
    let features = Mat::<f64>::from_fn(n_samples, n_features, |r, c| (r + c) as f64 * 0.5);
    let knots: Vec<Vec<f64>> = (0..n_features)
        .map(|p_idx| {
            let feature_col: Vec<f64> = (0..n_samples)
                .map(|r_idx| features[(r_idx, p_idx)])
                .collect();
            generate_quantile_knots(&feature_col, 5, 3)
        })
        .collect();
    StructuredDesignMatrix::new(features.as_ref(), &knots, degree)
}

fn test_helper_to_dense(matrix: &StructuredDesignMatrix) -> Mat<f64> {
    let mut dense = Mat::<f64>::zeros(matrix.n_samples, matrix.n_basis_total);
    let n_basis_active = matrix.degree + 1;

    for r_idx in 0..matrix.n_samples {
        dense[(r_idx, 0)] = 1.0;
    }

    for p_idx in 0..matrix.n_features {
        let col_offset = matrix.feature_offsets[p_idx];
        for r_idx in 0..matrix.n_samples {
            let flat_idx = r_idx * matrix.n_features + p_idx;
            let start_basis = matrix.basis_starts[flat_idx];
            let val_offset = flat_idx * n_basis_active;

            for k_idx in 0..n_basis_active {
                let col = col_offset + start_basis + k_idx;
                dense[(r_idx, col)] = matrix.basis_values[val_offset + k_idx];
            }
        }
    }

    dense
}

#[test]
fn test_matrix_shapes_and_dimensions() {
    let n_samples = 12;
    let n_features = 3;
    let degree = 2;

    let matrix = setup_test_matrix(n_samples, n_features, degree);
    let dense_ref = test_helper_to_dense(&matrix);

    assert_eq!(matrix.n_samples, n_samples);
    assert_eq!(matrix.n_features, n_features);
    assert_eq!(matrix.degree, degree);

    assert_eq!(matrix.basis_starts.len(), n_samples * n_features);
    assert_eq!(
        matrix.basis_values.len(),
        n_samples * n_features * (degree + 1)
    );
    assert_eq!(matrix.feature_offsets.len(), n_features);

    assert_eq!(dense_ref.nrows(), n_samples);
    assert_eq!(dense_ref.ncols(), matrix.n_basis_total);

    for r_idx in 0..n_samples {
        assert_eq!(dense_ref[(r_idx, 0)], 1.0);
    }
}

#[test]
fn test_forward_multiply_mathematical_correctness() {
    let matrix = setup_test_matrix(15, 4, 3);
    let dense_ref = test_helper_to_dense(&matrix);
    let coef: Vec<f64> = (0..matrix.n_basis_total)
        .map(|b_idx| b_idx as f64 * 0.1)
        .collect();

    let mut out_preds = vec![0.0; matrix.n_samples];
    matrix.multiply(&coef, &mut out_preds);

    for r_idx in 0..matrix.n_samples {
        let mut expected_val = 0.0;
        for b_idx in 0..matrix.n_basis_total {
            expected_val += dense_ref[(r_idx, b_idx)] * coef[b_idx];
        }
        assert!((out_preds[r_idx] - expected_val).abs() < 1e-9);
    }
}

#[test]
fn test_transpose_multiply_mathematical_correctness() {
    let matrix = setup_test_matrix(20, 2, 2);
    let dense_ref = test_helper_to_dense(&matrix);
    let gradients: Vec<f64> = (0..matrix.n_samples)
        .map(|r_idx| (r_idx as f64 * 0.05).sin())
        .collect();

    let mut out_coef_grads = vec![0.0; matrix.n_basis_total];
    matrix.transpose_multiply(&gradients, &mut out_coef_grads);

    for b_idx in 0..matrix.n_basis_total {
        let mut expected_grad = 0.0;
        for r_idx in 0..matrix.n_samples {
            expected_grad += dense_ref[(r_idx, b_idx)] * gradients[r_idx];
        }
        assert!((out_coef_grads[b_idx] - expected_grad).abs() < 1e-9);
    }
}

#[test]
fn test_compute_btb_hessian_correctness() {
    let matrix = setup_test_matrix(30, 3, 2);
    let dense_ref = test_helper_to_dense(&matrix);
    let btb_optimized = matrix.compute_btb();

    let n_total = matrix.n_basis_total;
    let n_features = matrix.n_features;

    for b1_idx in 0..n_total {
        for b2_idx in 0..n_total {
            let is_same_block = if b1_idx == 0 || b2_idx == 0 {
                true
            } else {
                let mut feat_of_b1 = 0;
                let mut feat_of_b2 = 0;
                for p_idx in 0..n_features {
                    let start = matrix.feature_offsets[p_idx];
                    let end = if p_idx + 1 < n_features {
                        matrix.feature_offsets[p_idx + 1]
                    } else {
                        n_total
                    };
                    if b1_idx >= start && b1_idx < end {
                        feat_of_b1 = p_idx;
                    }
                    if b2_idx >= start && b2_idx < end {
                        feat_of_b2 = p_idx;
                    }
                }
                feat_of_b1 == feat_of_b2
            };

            if is_same_block {
                let mut expected_cell = 0.0;
                for r_idx in 0..matrix.n_samples {
                    expected_cell += dense_ref[(r_idx, b1_idx)] * dense_ref[(r_idx, b2_idx)];
                }
                let actual_cell = btb_optimized[(b1_idx, b2_idx)];
                let delta = (actual_cell - expected_cell).abs();

                assert!(
                    delta < 1e-9,
                    "Mismatch at ({}, {}): actual={}, expected={}, delta={}",
                    b1_idx,
                    b2_idx,
                    actual_cell,
                    expected_cell,
                    delta
                );
            }
        }
    }
}

#[test]
fn test_intercept_only_coefficients() {
    let matrix = setup_test_matrix(10, 5, 3);
    let mut coef = vec![0.0; matrix.n_basis_total];
    coef[0] = 3.5;

    let mut out_preds = vec![0.0; matrix.n_samples];
    matrix.multiply(&coef, &mut out_preds);

    for pred in out_preds.iter() {
        assert!((pred - 3.5).abs() < 1e-9);
    }
}
