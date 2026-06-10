/// Evaluates B-spline basis functions using De Boor's recursion.
/// $B_{i,0}(x) = 1$ if $t_i \le x < t_{i+1}$, else $0$.
/// $B_{i,d}(x) = \frac{x - t_i}{t_{i+d} - t_i} B_{i,d-1}(x) + \frac{t_{i+d+1} - x}{t_{i+d+1} - t_{i+1}} B_{i+1,d-1}(x)$.
pub fn evaluate_bspline(x_raw: f64, knots: &[f64], degree: usize, out_basis: &mut [f64]) -> usize {
    let n_knots = knots.len();
    let required_buffer_len = degree + 1;

    assert!(
        out_basis.len() >= required_buffer_len,
        "out_basis length must be at least degree + 1 (expected {}, got {})",
        required_buffer_len,
        out_basis.len()
    );

    if n_knots < 2 * required_buffer_len {
        out_basis[..required_buffer_len].fill(0.0);
        return 0;
    }

    // Clamp $x$ to the valid domain $[t_d, t_{n-d-1}]$
    let min_knot = knots[degree];
    let max_knot = knots[n_knots - degree - 1];
    let x = x_raw.clamp(min_knot, max_knot);

    // Locate the knot span index $j$ such that $t_j \le x < t_{j+1}$
    let mut span_idx = match knots.binary_search_by(|val| val.total_cmp(&x)) {
        Ok(idx) => {
            // Handle multiple knots at the same location to find the rightmost index
            let mut search_idx = idx;
            while search_idx + 1 < n_knots && knots[search_idx + 1] == x {
                search_idx += 1;
            }
            search_idx
        }
        Err(idx) => {
            // Standard binary search result for value not in list
            if idx == 0 { 0 } else { idx - 1 }
        }
    };

    // Ensure span index is within valid range for $d$-degree basis
    if span_idx < degree {
        span_idx = degree;
    } else if span_idx >= n_knots - degree - 1 {
        span_idx = n_knots - degree - 2;
    }

    // Base case: 0-degree basis $B_{i,0}(x) = 1$
    out_basis[0] = 1.0;
    out_basis[1..=degree].fill(0.0);

    // De Boor's recursion: compute degree d_idx from d_idx-1
    for d_idx in 1..=degree {
        let mut prev_val = 0.0;
        for lb_idx in 0..d_idx {
            // Indices for the two lower-degree basis functions contributing to $B_{lb_idx,d_idx}$
            let knot_left = knots[span_idx + 1 + lb_idx - d_idx];
            let knot_right = knots[span_idx + 1 + lb_idx];

            // Convex combination coefficient $\alpha = \frac{x - t_i}{t_{i+k} - t_i}$
            let alpha = if knot_right == knot_left {
                0.0
            } else {
                (x - knot_left) / (knot_right - knot_left)
            };

            // Update in-place: $B_{i,k} = (1-\alpha)B_{i,k-1} + \alpha_{prev}B_{i-1,k-1}$
            let current_val = out_basis[lb_idx];
            out_basis[lb_idx] = prev_val + (1.0 - alpha) * current_val;
            prev_val = alpha * current_val;
        }
        // Last term for the current degree d_idx
        out_basis[d_idx] = prev_val;
    }

    span_idx - degree
}
