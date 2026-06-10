/// Huber-smoothed Pinball Loss for differentiable quantile regression.
/// $L_\epsilon(r, q) = \frac{r^2}{2\epsilon}$ if $|r| \le \epsilon$, else pinball loss with bias.
#[inline(always)]
pub fn huber_pinball_loss(r: f64, quantile: f64, epsilon: f64) -> f64 {
    let lower = -epsilon * (1.0 - quantile);
    let upper = epsilon * quantile;

    if r < lower {
        (quantile - 1.0) * r - 0.5 * epsilon * (1.0 - quantile).powi(2)
    } else if r > upper {
        quantile * r - 0.5 * epsilon * quantile.powi(2)
    } else {
        r.powi(2) / (2.0 * epsilon)
    }
}

/// Gradient of Huber-smoothed Pinball Loss $\nabla_r L_\epsilon(r, q)$.
#[inline(always)]
pub fn huber_pinball_grad(r: f64, quantile: f64, epsilon: f64) -> f64 {
    let lower = -epsilon * (1.0 - quantile);
    let upper = epsilon * quantile;

    if r < lower {
        quantile - 1.0
    } else if r > upper {
        quantile
    } else {
        r / epsilon
    }
}

/// Hessian of Huber-smoothed Pinball Loss $\nabla_r^2 L_\epsilon(r, q)$.
#[inline(always)]
pub fn huber_pinball_hess(r: f64, quantile: f64, epsilon: f64) -> f64 {
    let lower = -epsilon * (1.0 - quantile);
    let upper = epsilon * quantile;

    if r >= lower && r <= upper {
        1.0 / epsilon
    } else {
        0.0
    }
}
