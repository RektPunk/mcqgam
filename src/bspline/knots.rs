/// Generates a quantile-based knot vector for B-spline expansion.
pub fn generate_quantile_knots(
    samples: &[f64],
    n_internal_knots: usize,
    degree: usize,
) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut valid_samples: Vec<f64> = samples.iter().filter(|&x| x.is_finite()).copied().collect();

    if valid_samples.is_empty() {
        return Vec::new();
    }
    let n_valid_samples = valid_samples.len();

    let mut target_indices = Vec::with_capacity(n_internal_knots + 2);
    target_indices.push(0);

    if n_internal_knots > 0 && n_valid_samples > 1 {
        let n_intervals = (n_internal_knots + 1) as f64;
        let max_idx = (n_valid_samples - 1) as f64;

        for k in 1..=n_internal_knots {
            let quantile = k as f64 / n_intervals;
            let idx = (quantile * max_idx).round() as usize;
            target_indices.push(idx);
        }
    }

    target_indices.push(n_valid_samples.saturating_sub(1));
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

    let mut knots = Vec::with_capacity(target_indices.len().saturating_sub(2) + 2 * (degree + 1));
    knots.resize(degree + 1, min_value);
    let internal_bounds = target_indices.len().saturating_sub(1);
    if internal_bounds > 1 {
        knots.extend(
            target_indices[1..internal_bounds]
                .iter()
                .map(|&idx| valid_samples[idx]),
        );
    }

    knots.resize(knots.len() + degree + 1, max_value);

    knots
}
