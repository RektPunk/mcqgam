use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::bspline::basis::evaluate_bspline;

const CHUNK_SIZE: usize = 1024;

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
    pub basis_starts: Vec<usize>,
    /// Evaluated values of active basis functions ($B_{d,k}$).
    pub basis_values: Vec<f64>,
    /// Feature-wise offsets in global coefficient vector.
    pub feature_offsets: Vec<usize>,
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
            // Basis functions $M = K - (d + 1)$ where $K$ is knot count
            let basis_count = knots.len() - n_basis_active;
            feature_offsets.push(current_offset);
            current_offset += basis_count;
        }

        let mut basis_starts = vec![0; n_samples * n_features];
        let mut basis_values = vec![0.0; n_samples * n_features * n_basis_active];

        // Parallelize across row chunks for faster construction
        let first_indices_chunks = basis_starts.par_chunks_mut(CHUNK_SIZE * n_features);
        let basis_values_chunks =
            basis_values.par_chunks_mut(CHUNK_SIZE * n_features * n_basis_active);

        first_indices_chunks
            .zip(basis_values_chunks)
            .enumerate()
            .for_each(|(c_idx, (idx_chunk, val_chunk))| {
                let start_row = c_idx * CHUNK_SIZE;
                let actual_chunk_rows = idx_chunk.len() / n_features;
                let mut basis_buffer = vec![0.0; n_basis_active];

                for lr_idx in 0..actual_chunk_rows {
                    let r_idx = start_row + lr_idx;
                    for p_idx in 0..n_features {
                        let x_val = x[(r_idx, p_idx)];
                        // Compute local non-zero basis values and their global start index
                        let start_idx =
                            evaluate_bspline(x_val, &knots_vec[p_idx], degree, &mut basis_buffer);

                        idx_chunk[lr_idx * n_features + p_idx] = start_idx;
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

    /// Computes $y = B \cdot \text{coef}$ where $y_i = \text{coef}_0 + \sum_{d=1}^D \sum_{k=0}^d B_{d,k}(x_{i,d}) \text{coef}_{d,k}$.
    pub fn multiply(&self, coefficients: &[f64], y_pred: &mut [f64]) {
        assert_eq!(
            coefficients.len(),
            self.n_basis_total,
            "Coefficients size mismatch"
        );
        assert_eq!(y_pred.len(), self.n_samples, "Predictions size mismatch");
        let n_basis_active = self.degree + 1;
        let n_features = self.n_features;
        let intercept = coefficients[0];

        y_pred
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(c_idx, chunk)| {
                let start_row = c_idx * CHUNK_SIZE;
                for (lr_idx, pred_val) in chunk.iter_mut().enumerate() {
                    let r_idx = start_row + lr_idx;
                    let mut sum = intercept;
                    let row_base = r_idx * n_features;

                    for p_idx in 0..n_features {
                        let flat_idx = row_base + p_idx;
                        let b_off = self.basis_starts[flat_idx];
                        let val_off = flat_idx * n_basis_active;
                        let col_off = self.feature_offsets[p_idx];

                        for k_idx in 0..n_basis_active {
                            sum += self.basis_values[val_off + k_idx]
                                * coefficients[col_off + b_off + k_idx];
                        }
                    }
                    *pred_val = sum;
                }
            });
    }

    /// Computes $z = B^T \cdot \text{grads}$ for gradient backpropagation.
    pub fn transpose_multiply(&self, grads: &[f64], out_coef_grads: &mut [f64]) {
        assert_eq!(grads.len(), self.n_samples, "Gradients size mismatch");
        assert_eq!(
            out_coef_grads.len(),
            self.n_basis_total,
            "Coefficient gradients size mismatch"
        );

        let n_basis_active = self.degree + 1;
        let total_grads = grads
            .par_chunks(CHUNK_SIZE)
            .enumerate()
            .fold(
                || vec![0.0; self.n_basis_total],
                |mut acc, (c_idx, grad_chunk)| {
                    let start_row = c_idx * CHUNK_SIZE;
                    for (lr_idx, &g_i) in grad_chunk.iter().enumerate() {
                        let r_idx = start_row + lr_idx;
                        acc[0] += g_i;
                        for p_idx in 0..self.n_features {
                            let flat = r_idx * self.n_features + p_idx;
                            let b_off = self.basis_starts[flat];
                            let val_off = flat * n_basis_active;
                            let col_off = self.feature_offsets[p_idx];

                            for k_idx in 0..n_basis_active {
                                acc[col_off + b_off + k_idx] +=
                                    self.basis_values[val_off + k_idx] * g_i;
                            }
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

    /// Computes $B^T B$ utilizing the sparse structure of $B$.
    pub fn compute_btb(&self) -> Mat<f64> {
        let n_basis_active = self.degree + 1;
        let n_basis_total = self.n_basis_total;

        let btb_vec = (0..self.n_samples)
            .into_par_iter()
            .chunks(CHUNK_SIZE)
            .fold(
                || vec![0.0; n_basis_total * n_basis_total],
                |mut acc, chunk| {
                    for r_idx in chunk {
                        acc[0] += 1.0;

                        for p_idx in 0..self.n_features {
                            let flat = r_idx * self.n_features + p_idx;
                            let b_off = self.basis_starts[flat];
                            let val_off = flat * n_basis_active;
                            let col_off = self.feature_offsets[p_idx];

                            for k_idx in 0..n_basis_active {
                                let val = self.basis_values[val_off + k_idx];
                                let idx = col_off + b_off + k_idx;
                                acc[idx] += val;
                                acc[idx * n_basis_total] += val;
                            }

                            for k1_idx in 0..n_basis_active {
                                let v1 = self.basis_values[val_off + k1_idx];
                                let idx1 = col_off + b_off + k1_idx;
                                for k2_idx in k1_idx..n_basis_active {
                                    let val = v1 * self.basis_values[val_off + k2_idx];
                                    let idx2 = col_off + b_off + k2_idx;
                                    let pos = idx1 * n_basis_total + idx2;
                                    acc[pos] += val;
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

        Mat::from_fn(n_basis_total, n_basis_total, |r_idx, b_idx| {
            btb_vec[b_idx * n_basis_total + r_idx]
        })
    }
}
