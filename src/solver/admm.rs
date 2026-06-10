use faer::{Mat, Side, prelude::*};
use rayon::prelude::*;

use crate::bspline::matrix::StructuredDesignMatrix;
use crate::solver::objective::huber_pinball_grad;
use crate::solver::pava::pava;

/// Configuration for the ADMM solver.
pub struct AdmmConfig {
    /// Maximum number of ADMM iterations.
    pub max_iter: usize,
    /// Maximum inner Newton iterations for primal update.
    pub max_inner_iter: usize,
    /// Augmented Lagrangian penalty parameter ($\rho$).
    pub rho: f64,
    /// $L_2$ regularization strength ($\lambda$).
    pub lambda: f64,
    /// Huber-smoothing parameter ($\epsilon$).
    pub epsilon: f64,
    /// Absolute convergence tolerance.
    pub atol: f64,
    /// Relative convergence tolerance.
    pub rtol: f64,
}

impl AdmmConfig {
    pub fn new(
        max_iter: usize,
        max_inner_iter: usize,
        rho: f64,
        lambda: f64,
        epsilon: f64,
        atol: f64,
        rtol: f64,
    ) -> Self {
        Self {
            max_iter,
            max_inner_iter,
            rho,
            lambda,
            epsilon,
            atol,
            rtol,
        }
    }
}

/// Result of the ADMM optimization.
pub struct AdmmResult {
    /// Fitted coefficient matrix $\text{coef} \in \mathbb{R}^{K \times M}$.
    pub coeffs: Mat<f64>,
    /// Number of total iterations performed.
    pub iterations: usize,
    /// Convergence status flag.
    pub converged: bool,
}

/// Thread-local workspace for a single quantile optimization.
struct QuantileWorkspace {
    /// Local prediction buffer ($\hat{y} = B \cdot \text{coef}$).
    y_pred: Vec<f64>,
    /// Huber loss gradients with respect to residuals ($\nabla_r L$).
    residual_gradients: Vec<f64>,
    /// Gradients with respect to coefficients ($B^T \nabla_r L$).
    coeff_grads: Vec<f64>,
    /// Current coefficient state for inner iterations.
    current_coef: Vec<f64>,
}

impl QuantileWorkspace {
    fn new(n_samples: usize, n_basis_total: usize) -> Self {
        Self {
            y_pred: vec![0.0; n_samples],
            residual_gradients: vec![0.0; n_samples],
            coeff_grads: vec![0.0; n_basis_total],
            current_coef: vec![0.0; n_basis_total],
        }
    }
}

/// Solves $\min_{\text{coef}} L(y, B \cdot \text{coef}) + \frac{\lambda}{2}\|\text{coef}\|_2^2$ subject to monotonicity.
/// Uses ADMM for constrained optimization.
pub fn solve_admm(
    y: ColRef<f64>,
    design_matrix: &StructuredDesignMatrix,
    quantiles: &[f64],
    config: &AdmmConfig,
) -> AdmmResult {
    let n_samples = y.nrows();
    assert_eq!(
        n_samples, design_matrix.n_samples,
        "Observations size mismatch"
    );

    let n_basis_total = design_matrix.n_basis_total;
    let n_quantiles = quantiles.len();

    let mut coef = Mat::<f64>::zeros(n_basis_total, n_quantiles);
    let mut monotone_coef = Mat::<f64>::zeros(n_basis_total, n_quantiles);
    let mut dual_var = Mat::<f64>::zeros(n_basis_total, n_quantiles);

    let mut current_rho = config.rho;
    let mut converged = false;
    let mut final_iter = 0;

    let btb = design_matrix.compute_btb();
    let factorize_hessian = |rho: f64| {
        let mut hessian = btb.clone();
        for b_idx in 0..n_basis_total {
            // Augmented Hessian $H = B^T B + (\lambda + \rho)I$
            hessian[(b_idx, b_idx)] += config.lambda + rho;
        }
        hessian
            .llt(Side::Lower)
            .expect("Hessian factorization failed")
    };

    let mut llt_factorization = factorize_hessian(current_rho);

    // Weights for CRPS approximation: $\sum w_i = M$
    let mut weights = vec![1.0; n_quantiles];
    if n_quantiles > 1 {
        for q_idx in 0..n_quantiles {
            // Trapezoidal-like rule for quantile spacing
            let w = if q_idx == 0 {
                (quantiles[0] + quantiles[1]) / 2.0
            } else if q_idx == n_quantiles - 1 {
                1.0 - (quantiles[q_idx - 1] + quantiles[q_idx]) / 2.0
            } else {
                (quantiles[q_idx + 1] - quantiles[q_idx - 1]) / 2.0
            };
            weights[q_idx] = w * n_quantiles as f64;
        }
    }

    let mut workspaces: Vec<QuantileWorkspace> = (0..n_quantiles)
        .map(|_| QuantileWorkspace::new(n_samples, n_basis_total))
        .collect();

    for iter_idx in 0..config.max_iter {
        final_iter = iter_idx;
        let z_prev = monotone_coef.clone();

        // Primal update: $\text{coef}^{k+1} = \text{argmin}_{\text{coef}} \mathcal{L}(\text{coef}, z^k, u^k)$
        workspaces
            .par_iter_mut()
            .enumerate()
            .for_each(|(q_idx, ws)| {
                let quantile = quantiles[q_idx];
                let weight = weights[q_idx];

                for b_idx in 0..n_basis_total {
                    ws.current_coef[b_idx] = coef[(b_idx, q_idx)];
                }

                for _ in 0..config.max_inner_iter {
                    // Compute predictions $y = B \cdot \text{coef}$
                    design_matrix.multiply(&ws.current_coef, &mut ws.y_pred);

                    // Compute Huber-smoothed pinball gradients $\nabla L$
                    let chunk_size = 2048;
                    ws.residual_gradients
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(c_idx, chunk)| {
                            let start = c_idx * chunk_size;
                            for (lr_idx, g_val) in chunk.iter_mut().enumerate() {
                                let r_idx = start + lr_idx;
                                let r = y[r_idx] - ws.y_pred[r_idx];
                                *g_val = huber_pinball_grad(r, quantile, config.epsilon);
                            }
                        });

                    // Backpropagate gradients to coefficients $B^T \nabla L$
                    design_matrix.transpose_multiply(&ws.residual_gradients, &mut ws.coeff_grads);

                    let mut grad_col = Col::<f64>::zeros(n_basis_total);
                    for b_idx in 0..n_basis_total {
                        // Total gradient $\nabla f(\text{coef}) = -B^T \nabla L + \lambda \text{coef} + \rho (\text{coef} - z + u)$
                        grad_col[b_idx] = -ws.coeff_grads[b_idx] * weight
                            + config.lambda * ws.current_coef[b_idx]
                            + current_rho
                                * (ws.current_coef[b_idx] - monotone_coef[(b_idx, q_idx)]
                                    + dual_var[(b_idx, q_idx)]);
                    }

                    // Newton step: $\Delta \text{coef} = -H^{-1} \nabla f$
                    let search_direction = llt_factorization.solve(-&grad_col);
                    for b_idx in 0..n_basis_total {
                        ws.current_coef[b_idx] += search_direction[b_idx];
                    }

                    // Convergence check for inner Newton iterations
                    if search_direction.norm_max() < 1e-7 {
                        break;
                    }
                }
            });

        for (q_idx, ws) in workspaces.iter().enumerate() {
            for b_idx in 0..n_basis_total {
                coef[(b_idx, q_idx)] = ws.current_coef[b_idx];
            }
        }

        // Auxiliary update: $z^{k+1} = \text{argmin}_z \frac{\rho}{2}\|\text{coef}^{k+1} - z + u^k\|_2^2$ s.t. $z_{i,m} \le z_{i,m+1}$
        let updated_monotone_zs: Vec<Vec<f64>> = (0..n_basis_total)
            .into_par_iter()
            .map(|b_idx| {
                let mut v_targets = vec![0.0; n_quantiles];
                let mut z_projection = vec![0.0; n_quantiles];
                for q_idx in 0..n_quantiles {
                    v_targets[q_idx] = coef[(b_idx, q_idx)] + dual_var[(b_idx, q_idx)];
                }
                // Isotonic regression via PAVA
                pava(&v_targets, &mut z_projection);
                z_projection
            })
            .collect();

        for (b_idx, z_vec) in updated_monotone_zs.into_iter().enumerate() {
            for (q_idx, &val) in z_vec.iter().enumerate() {
                monotone_coef[(b_idx, q_idx)] = val;
            }
        }

        // Dual update: $u^{k+1} = u^k + \text{coef}^{k+1} - z^{k+1}$
        dual_var += &coef - &monotone_coef;

        // Residuals for convergence: $r = \|\text{coef} - z\|$, $s = \rho \|z - z_{prev}\|$
        let primal_residual = (&coef - &monotone_coef).norm_max();
        let dual_residual = current_rho * (&monotone_coef - &z_prev).norm_max();

        if primal_residual < config.atol && dual_residual < config.atol {
            converged = true;
            break;
        }

        // Adaptive penalty scaling
        if iter_idx > 0 && iter_idx % 20 == 0 {
            if primal_residual > 5.0 * dual_residual {
                current_rho *= 1.5;
            } else if dual_residual > 5.0 * primal_residual {
                current_rho /= 1.5;
            }
            llt_factorization = factorize_hessian(current_rho);
        }
    }

    AdmmResult {
        coeffs: monotone_coef,
        iterations: final_iter + 1,
        converged,
    }
}
