/// Generates a quantile-based knot vector for B-spline expansion.
pub fn generate_quantile_knots(
    samples: &[f64],
    n_internal_knots: usize,
    degree: usize,
) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut valid_samples: Vec<f64> = samples.iter().copied().filter(|&x| !x.is_nan()).collect();
    if valid_samples.is_empty() {
        return Vec::new();
    }
    let n_valid_samples = valid_samples.len();

    let mut target_indices = Vec::with_capacity(n_internal_knots + 2);
    target_indices.push(0);
    target_indices.push(n_valid_samples - 1);

    if n_internal_knots > 0 && n_valid_samples > 1 {
        for t_idx in 1..=n_internal_knots {
            // Knot at quantile $q = \frac{k}{K+1}$
            let quantile = t_idx as f64 / (n_internal_knots + 1) as f64;
            let target_idx = (quantile * (n_valid_samples - 1) as f64) as usize;
            target_indices.push(target_idx);
        }
    }

    target_indices.sort_unstable();
    target_indices.dedup();

    // Multi-selection via partitioning for efficiency
    let mut current_start = 0;
    for &target_idx in &target_indices {
        if target_idx >= current_start {
            valid_samples[current_start..]
                .select_nth_unstable_by(target_idx - current_start, |a, b| a.total_cmp(b));
            current_start = target_idx + 1;
        }
    }

    let min_value = valid_samples[0];
    let max_value = valid_samples[n_valid_samples - 1];

    let mut internal_knots = Vec::with_capacity(n_internal_knots);
    if n_internal_knots > 0 && n_valid_samples > 1 {
        for t_idx in 1..=n_internal_knots {
            let quantile = t_idx as f64 / (n_internal_knots + 1) as f64;
            let r_idx = (quantile * (n_valid_samples - 1) as f64) as usize;
            internal_knots.push(valid_samples[r_idx]);
        }
    }

    internal_knots.sort_by(|a, b| a.total_cmp(b));
    internal_knots.dedup();

    // Augmented knot vector with $d+1$ boundary knots
    let mut knots = Vec::with_capacity(internal_knots.len() + 2 * (degree + 1));
    knots.resize(degree + 1, min_value);
    knots.extend(internal_knots);
    knots.resize(knots.len() + degree + 1, max_value);

    knots
}
