use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::bspline::basis::evaluate_bspline;

const CHUNK_SIZE: usize = 1024;

struct ActiveBasesView<'a> {
    coef_start: usize,
    basis_values: &'a [f64],
}

/// Compressed B-spline design matrix $B$.
pub struct StructuredDesignMatrix {
    /// Number of data samples ($N$).
    pub n_samples: usize,
    /// Number of input features ($D$).
    pub n_features: usize,
    /// Total number of basis functions including global intercept ($K$).
    pub n_basis_total: usize,
    /// B-spline degree ($d$).
    pub degree: usize,
    /// Start index of non-zero basis functions for each $(i, d)$ pair.
    pub basis_starts: Vec<u32>,
    /// Evaluated values of active basis functions ($B_{d,k}$).
    pub basis_values: Vec<f64>,
    /// Feature-wise offsets in global coefficient vector.
    pub feature_offsets: Vec<u32>,
}

impl StructuredDesignMatrix {
    /// Constructs design matrix $B$ from features and knots.
    pub fn new(x: MatRef<f64>, knots_vec: &[Vec<f64>], degree: usize) -> Self {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_basis_active = degree + 1;

        let mut feature_offsets = Vec::with_capacity(n_features);
        let mut current_offset = 1;

        for knots in knots_vec {
            let basis_count = knots.len() - n_basis_active;
            feature_offsets.push(current_offset as u32);
            current_offset += basis_count;
        }

        let mut basis_starts = vec![0; n_samples * n_features];
        let mut basis_values = vec![0.0; n_samples * n_features * n_basis_active];

        basis_starts
            .par_chunks_mut(CHUNK_SIZE * n_features)
            .zip(basis_values.par_chunks_mut(CHUNK_SIZE * n_features * n_basis_active))
            .enumerate()
            .for_each(|(c_idx, (idx_chunk, val_chunk))| {
                let start_row = c_idx * CHUNK_SIZE;
                let actual_chunk_rows = idx_chunk.len() / n_features;
                let mut basis_buffer = vec![0.0; n_basis_active];

                for lr_idx in 0..actual_chunk_rows {
                    let r_idx = start_row + lr_idx;
                    for p_idx in 0..n_features {
                        let x_val = x[(r_idx, p_idx)];
                        let start_idx =
                            evaluate_bspline(x_val, &knots_vec[p_idx], degree, &mut basis_buffer);

                        idx_chunk[lr_idx * n_features + p_idx] = start_idx as u32;
                        let val_offset = (lr_idx * n_features + p_idx) * n_basis_active;
                        val_chunk[val_offset..val_offset + n_basis_active]
                            .copy_from_slice(&basis_buffer);
                    }
                }
            });

        Self {
            n_samples,
            n_features,
            n_basis_total: current_offset,
            degree,
            basis_starts,
            basis_values,
            feature_offsets,
        }
    }

    #[inline(always)]
    fn view_basis(&self, r_idx: usize, p_idx: usize) -> ActiveBasesView<'_> {
        let n_basis_active = self.degree + 1;
        let flat_idx = r_idx * self.n_features + p_idx;
        let val_off = flat_idx * n_basis_active;
        let basis_start = self.basis_starts[flat_idx] as usize;
        let feature_offset = self.feature_offsets[p_idx] as usize;

        ActiveBasesView {
            coef_start: feature_offset + basis_start,
            basis_values: &self.basis_values[val_off..val_off + n_basis_active],
        }
    }

    /// Computes y where y_i = coef_0 + \sum_{d=1}^D \sum_{k=0}^d B_{d,k}(x_{i,d}) coef_{d,k}.
    pub fn multiply(&self, coefficients: &[f64], y_pred: &mut [f64]) {
        assert_eq!(
            coefficients.len(),
            self.n_basis_total,
            "Coefficients size mismatch"
        );
        assert_eq!(y_pred.len(), self.n_samples, "Predictions size mismatch");
        let n_basis_active = self.degree + 1;
        let intercept = coefficients[0];

        y_pred
            .par_iter_mut()
            .enumerate()
            .for_each(|(r_idx, pred_val)| {
                let mut sum = intercept;
                for p_idx in 0..self.n_features {
                    let view = self.view_basis(r_idx, p_idx);
                    for k_idx in 0..n_basis_active {
                        sum += view.basis_values[k_idx] * coefficients[view.coef_start + k_idx];
                    }
                }
                *pred_val = sum;
            });
    }

    /// Computes z = B^T grads for gradient backpropagation.
    pub fn transpose_multiply(&self, grads: &[f64], out_coef_grads: &mut [f64]) {
        assert_eq!(grads.len(), self.n_samples, "Gradients size mismatch");
        assert_eq!(
            out_coef_grads.len(),
            self.n_basis_total,
            "Coefficient gradients size mismatch"
        );

        let n_basis_active = self.degree + 1;
        let total_grads = (0..self.n_samples)
            .into_par_iter()
            .fold(
                || vec![0.0; self.n_basis_total],
                |mut acc, r_idx| {
                    let g_i = grads[r_idx];
                    acc[0] += g_i;

                    for p_idx in 0..self.n_features {
                        let view = self.view_basis(r_idx, p_idx);
                        for k_idx in 0..n_basis_active {
                            acc[view.coef_start + k_idx] += view.basis_values[k_idx] * g_i;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; self.n_basis_total],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );

        out_coef_grads.copy_from_slice(&total_grads);
    }

    /// Computes B^TB utilizing the sparse structure of B.
    pub fn compute_btb(&self) -> Mat<f64> {
        let n_basis_active = self.degree + 1;
        let n_basis_total = self.n_basis_total;

        let btb_vec = (0..self.n_samples)
            .into_par_iter()
            .fold(
                || vec![0.0; n_basis_total * n_basis_total],
                |mut acc, r_idx| {
                    acc[0] += 1.0;

                    for p1 in 0..self.n_features {
                        let view1 = self.view_basis(r_idx, p1);

                        for k1 in 0..n_basis_active {
                            let idx1 = view1.coef_start + k1;
                            let v1 = view1.basis_values[k1];
                            acc[idx1] += v1;
                            acc[idx1 * n_basis_total] += v1;
                        }

                        for p2 in p1..self.n_features {
                            let view2 = self.view_basis(r_idx, p2);
                            let is_same_feature = p1 == p2;

                            for k1 in 0..n_basis_active {
                                let idx1 = view1.coef_start + k1;
                                let v1 = view1.basis_values[k1];

                                let k2_start = if is_same_feature { k1 } else { 0 };

                                for k2 in k2_start..n_basis_active {
                                    let idx2 = view2.coef_start + k2;
                                    let val = v1 * view2.basis_values[k2];

                                    acc[idx1 * n_basis_total + idx2] += val;
                                    if idx1 != idx2 {
                                        acc[idx2 * n_basis_total + idx1] += val;
                                    }
                                }
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; n_basis_total * n_basis_total],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );

        Mat::from_fn(n_basis_total, n_basis_total, |r_idx, c_idx| {
            btb_vec[r_idx * n_basis_total + c_idx]
        })
    }
}
