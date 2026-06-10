/// Monotone Cubic Spline interpolation.
/// Ensures $f(q_i) \le f(q_{i+1})$ if $y_i \le y_{i+1}$.
pub struct MonotoneCubicSpline {
    /// Input quantile coordinates.
    quantiles: Vec<f64>,
    /// Predicted values at input quantiles.
    preds: Vec<f64>,
    /// Computed derivatives at each point ensuring monotonicity.
    derivatives: Vec<f64>,
}

impl MonotoneCubicSpline {
    pub fn new(quantiles: Vec<f64>, preds: Vec<f64>) -> Self {
        let n_points = quantiles.len();
        if n_points < 2 {
            return Self {
                quantiles,
                preds,
                derivatives: vec![0.0; n_points],
            };
        }

        let mut slopes = vec![0.0; n_points - 1];
        for q_idx in 0..n_points - 1 {
            // Finite difference slope $s_i = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}$
            slopes[q_idx] =
                (preds[q_idx + 1] - preds[q_idx]) / (quantiles[q_idx + 1] - quantiles[q_idx]);
        }

        let mut derivatives = vec![0.0; n_points];
        derivatives[0] = slopes[0];
        for q_idx in 1..n_points - 1 {
            // Average slope $d_i = \frac{s_{i-1} + s_i}{2}$
            derivatives[q_idx] = (slopes[q_idx - 1] + slopes[q_idx]) / 2.0;
        }
        derivatives[n_points - 1] = slopes[n_points - 2];

        // Fritsch-Butland algorithm for monotonicity
        for q_idx in 0..n_points - 1 {
            if slopes[q_idx] == 0.0 {
                // Flat region requires zero derivatives for monotonicity
                derivatives[q_idx] = 0.0;
                derivatives[q_idx + 1] = 0.0;
            } else {
                let alpha = derivatives[q_idx] / slopes[q_idx];
                let beta = derivatives[q_idx + 1] / slopes[q_idx];
                let squared_dist = alpha * alpha + beta * beta;
                if squared_dist > 9.0 {
                    // Necessary and sufficient condition: scale $(d_i, d_{i+1})$ to lie inside $R$
                    let tau = 3.0 / squared_dist.sqrt();
                    derivatives[q_idx] = tau * alpha * slopes[q_idx];
                    derivatives[q_idx + 1] = tau * beta * slopes[q_idx];
                }
            }
        }

        Self {
            quantiles,
            preds,
            derivatives,
        }
    }

    /// Interpolates at $q$ using Hermite basis functions.
    /// $f(q) = y_i h_{00}(t) + h d_i h_{10}(t) + y_{i+1} h_{01}(t) + h d_{i+1} h_{11}(t)$
    pub fn interpolate(&self, quantile: f64) -> f64 {
        let n_points = self.quantiles.len();
        if n_points == 0 {
            return 0.0;
        }
        if n_points == 1 {
            return self.preds[0];
        }

        // Linear extrapolation
        if quantile <= self.quantiles[0] {
            return self.preds[0] + self.derivatives[0] * (quantile - self.quantiles[0]);
        }
        if quantile >= self.quantiles[n_points - 1] {
            return self.preds[n_points - 1]
                + self.derivatives[n_points - 1] * (quantile - self.quantiles[n_points - 1]);
        }

        let q_idx = match self
            .quantiles
            .binary_search_by(|val| val.total_cmp(&quantile))
        {
            Ok(idx) => return self.preds[idx],
            Err(idx) => idx - 1,
        };

        let h = self.quantiles[q_idx + 1] - self.quantiles[q_idx];
        let t = (quantile - self.quantiles[q_idx]) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        // Cubic Hermite basis functions
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * self.preds[q_idx]
            + h10 * h * self.derivatives[q_idx]
            + h01 * self.preds[q_idx + 1]
            + h11 * h * self.derivatives[q_idx + 1]
    }
}
