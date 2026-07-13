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
        let mut derivatives = vec![0.0; n_points];
        let mut prev_slope = (preds[1] - preds[0]) / (quantiles[1] - quantiles[0]);
        derivatives[0] = prev_slope;

        for i in 1..n_points - 1 {
            let curr_slope = (preds[i + 1] - preds[i]) / (quantiles[i + 1] - quantiles[i]);
            if prev_slope * curr_slope <= 0.0 {
                derivatives[i] = 0.0;
            } else {
                derivatives[i] = (prev_slope + curr_slope) / 2.0;
            }
            prev_slope = curr_slope;
        }
        derivatives[n_points - 1] = prev_slope;

        for i in 0..n_points - 1 {
            let slope = (preds[i + 1] - preds[i]) / (quantiles[i + 1] - quantiles[i]);

            if slope == 0.0 {
                derivatives[i] = 0.0;
                derivatives[i + 1] = 0.0;
            } else {
                let alpha = derivatives[i] / slope;
                let beta = derivatives[i + 1] / slope;
                let squared_dist = alpha * alpha + beta * beta;

                if squared_dist > 9.0 {
                    let tau = 3.0 / squared_dist.sqrt();
                    derivatives[i] *= tau;
                    derivatives[i + 1] *= tau;
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
        if quantile.is_nan() {
            return f64::NAN;
        }

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
