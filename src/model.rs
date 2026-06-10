use faer::{ColRef, Mat, MatRef};
use rayon::prelude::*;

use crate::bspline::knots::generate_quantile_knots;
use crate::bspline::matrix::StructuredDesignMatrix;
use crate::interpolation::cubic::MonotoneCubicSpline;
use crate::solver::admm::{AdmmConfig, solve_admm};

/// Monotone Quantile Generalized Additive Model (MQGAM).
/// Fits non-crossing quantile curves $f(x, q_m) = B(x) \cdot \text{coeffs}_m$.
pub struct Regressor {
    /// Degree of B-spline basis functions ($d$).
    pub degree: usize,
    /// Number of internal knots for each feature.
    pub num_internal_knots: usize,
    /// Target quantiles to fit ($\tau_1 < \tau_2 < \dots < \tau_M$).
    pub quantiles: Vec<f64>,
    /// Knot vectors for each input feature.
    pub knots_per_feature: Vec<Vec<f64>>,
    /// Fitted model coefficients matrix ($K \times M$).
    pub coeffs: Option<Mat<f64>>,
}

impl Regressor {
    /// Initializes MQGAM with sorted target quantiles.
    pub fn new(degree: usize, num_internal_knots: usize, quantiles: Vec<f64>) -> Self {
        let mut sorted_quantiles = quantiles;
        sorted_quantiles.sort_by(|a, b| a.total_cmp(b));
        Self {
            degree,
            num_internal_knots,
            quantiles: sorted_quantiles,
            knots_per_feature: Vec::new(),
            coeffs: None,
        }
    }

    /// Fits model coefficients using ADMM solver.
    pub fn fit(&mut self, x: MatRef<f64>, y: ColRef<f64>, config: AdmmConfig) {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Generate knots based on feature quantiles
        self.knots_per_feature = (0..n_features)
            .map(|p_idx| {
                let feature_col: Vec<f64> = (0..n_samples).map(|r_idx| x[(r_idx, p_idx)]).collect();
                generate_quantile_knots(&feature_col, self.num_internal_knots, self.degree)
            })
            .collect();

        let design_matrix = StructuredDesignMatrix::new(x, &self.knots_per_feature, self.degree);

        let result = solve_admm(y, &design_matrix, &self.quantiles, &config);

        self.coeffs = Some(result.coeffs);
    }

    /// Predicts $\hat{y}$ for given quantiles. Interpolates if $q \notin$ quantiles.
    pub fn predict(&self, x: MatRef<f64>, quantiles: Vec<f64>) -> Mat<f64> {
        let design_matrix = StructuredDesignMatrix::new(x, &self.knots_per_feature, self.degree);
        let coeffs = self.coeffs.as_ref().expect("Model not fitted");
        let n_samples = design_matrix.n_samples;
        let n_base_quantile = coeffs.ncols();

        // Parallel prediction for base quantiles: $\hat{y}_m = B(x) \cdot \text{coeffs}_m$
        let preds_vec: Vec<Vec<f64>> = (0..n_base_quantile)
            .into_par_iter()
            .map(|q_idx| {
                let mut current_preds = vec![0.0; n_samples];
                let current_coef_vec: Vec<f64> = (0..design_matrix.n_basis_total)
                    .map(|b_idx| coeffs[(b_idx, q_idx)])
                    .collect();

                design_matrix.multiply(&current_coef_vec, &mut current_preds);
                current_preds
            })
            .collect();

        let mut quantile_preds = Mat::<f64>::zeros(n_samples, quantiles.len());
        for (q_idx, &quantile) in quantiles.iter().enumerate() {
            if let Some(lq_pos_idx) = self
                .quantiles
                .iter()
                .position(|&q| (q - quantile).abs() < 1e-9)
            {
                // Return pre-computed base quantile predictions
                for r_idx in 0..n_samples {
                    quantile_preds[(r_idx, q_idx)] = preds_vec[lq_pos_idx][r_idx];
                }
            } else {
                // Monotone cubic spline interpolation across base quantiles
                for r_idx in 0..n_samples {
                    let sample_q_values: Vec<f64> = (0..n_base_quantile)
                        .map(|lq_idx| preds_vec[lq_idx][r_idx])
                        .collect();

                    let interpolator =
                        MonotoneCubicSpline::new(self.quantiles.clone(), sample_q_values);
                    quantile_preds[(r_idx, q_idx)] = interpolator.interpolate(quantile);
                }
            }
        }

        quantile_preds
    }
}
