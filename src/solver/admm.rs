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
    /// Fitted coefficient matrix (K x M).
    pub coeffs: Mat<f64>,
    /// Number of total iterations performed.
    pub iterations: usize,
    /// Convergence status flag.
    pub converged: bool,
}

struct QuantileWorkspace {
    y_pred: Vec<f64>,
    residual_gradients: Vec<f64>,
    coeff_grads: Vec<f64>,
    current_coef: Vec<f64>,
    grad_col: Col<f64>,
}

impl QuantileWorkspace {
    fn new(n_samples: usize, n_basis_total: usize) -> Self {
        Self {
            y_pred: vec![0.0; n_samples],
            residual_gradients: vec![0.0; n_samples],
            coeff_grads: vec![0.0; n_basis_total],
            current_coef: vec![0.0; n_basis_total],
            grad_col: Col::<f64>::zeros(n_basis_total),
        }
    }
}

struct PavaWorkspace {
    v_targets: Vec<f64>,
    z_projection: Vec<f64>,
}

/// Solves constrained optimization using ADMM.
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
    let mut z_prev = Mat::<f64>::zeros(n_basis_total, n_quantiles);

    let mut current_rho = config.rho;
    let mut converged = false;
    let mut final_iter = 0;

    let btb = design_matrix.compute_btb();
    let factorize_hessian = |rho: f64| {
        let mut hessian = btb.clone();
        for b_idx in 0..n_basis_total {
            // Augmented Hessian H = B^T B + (\lambda + \rho)I
            hessian[(b_idx, b_idx)] += config.lambda + rho;
        }
        hessian
            .llt(Side::Lower)
            .expect("Hessian factorization failed")
    };

    let mut llt_factorization = factorize_hessian(current_rho);

    // Weights for CRPS approximation
    let mut weights = vec![1.0; n_quantiles];
    if n_quantiles > 1 {
        for q_idx in 0..n_quantiles {
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

    let mut pava_workspaces: Vec<PavaWorkspace> = (0..n_basis_total)
        .map(|_| PavaWorkspace {
            v_targets: vec![0.0; n_quantiles],
            z_projection: vec![0.0; n_quantiles],
        })
        .collect();

    for iter_idx in 0..config.max_iter {
        final_iter = iter_idx;
        z_prev.as_mut().copy_from(monotone_coef.as_ref());

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
                    // Compute predictions
                    design_matrix.multiply(&ws.current_coef, &mut ws.y_pred);

                    // Compute Huber-smoothed pinball gradients
                    for r_idx in 0..n_samples {
                        let r = y[r_idx] - ws.y_pred[r_idx];
                        ws.residual_gradients[r_idx] =
                            huber_pinball_grad(r, quantile, config.epsilon);
                    }

                    // Backpropagate negative gradients to coefficients
                    design_matrix.transpose_multiply(&ws.residual_gradients, &mut ws.coeff_grads);
                    for b_idx in 0..n_basis_total {
                        ws.grad_col[b_idx] = ws.coeff_grads[b_idx] * weight
                            - config.lambda * ws.current_coef[b_idx]
                            - current_rho
                                * (ws.current_coef[b_idx] - monotone_coef[(b_idx, q_idx)]
                                    + dual_var[(b_idx, q_idx)]);
                    }

                    // Newton step
                    llt_factorization.solve_in_place(&mut ws.grad_col);
                    let mut max_step: f64 = 0.0;
                    for b_idx in 0..n_basis_total {
                        let step = ws.grad_col[b_idx];
                        ws.current_coef[b_idx] += step;
                        max_step = max_step.max(step.abs());
                    }

                    if max_step < 1e-7 {
                        break;
                    }
                }
            });

        for (q_idx, ws) in workspaces.iter().enumerate() {
            for b_idx in 0..n_basis_total {
                coef[(b_idx, q_idx)] = ws.current_coef[b_idx];
            }
        }

        // Auxiliary update
        pava_workspaces
            .par_iter_mut()
            .enumerate()
            .for_each(|(b_idx, p_ws)| {
                for q_idx in 0..n_quantiles {
                    p_ws.v_targets[q_idx] = coef[(b_idx, q_idx)] + dual_var[(b_idx, q_idx)];
                }
                pava(&p_ws.v_targets, &mut p_ws.z_projection);
            });

        for (b_idx, p_ws) in pava_workspaces.iter().enumerate() {
            for q_idx in 0..n_quantiles {
                monotone_coef[(b_idx, q_idx)] = p_ws.z_projection[q_idx];
            }
        }

        // Compute residuals and convergence metrics
        let mut primal_residual: f64 = 0.0;
        let mut dual_residual: f64 = 0.0;
        let mut max_coef_norm: f64 = 0.0;
        let mut max_zt_norm: f64 = 0.0;
        let mut max_dual_norm: f64 = 0.0;
        for q_idx in 0..n_quantiles {
            for b_idx in 0..n_basis_total {
                let c_val = coef[(b_idx, q_idx)];
                let z_val = monotone_coef[(b_idx, q_idx)];
                let z_old = z_prev[(b_idx, q_idx)];

                dual_var[(b_idx, q_idx)] += c_val - z_val;
                let u_val = dual_var[(b_idx, q_idx)];

                primal_residual = primal_residual.max((c_val - z_val).abs());
                dual_residual = dual_residual.max((z_val - z_old).abs());

                max_coef_norm = max_coef_norm.max(c_val.abs());
                max_zt_norm = max_zt_norm.max(z_val.abs());
                max_dual_norm = max_dual_norm.max(u_val.abs());
            }
        }

        dual_residual *= current_rho;

        let eps_primal = config.atol + config.rtol * max_coef_norm.max(max_zt_norm);
        let eps_dual = config.atol + config.rtol * current_rho * max_dual_norm;

        if primal_residual < eps_primal && dual_residual < eps_dual {
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
