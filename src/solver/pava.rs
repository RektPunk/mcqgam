/// Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
/// Solves $\min \sum (x_i - y_i)^2$ subject to $x_1 \le x_2 \le \dots \le x_n$.
pub fn pava(input: &[f64], output: &mut [f64]) {
    let n_elements = input.len();
    if n_elements == 0 {
        return;
    }
    if n_elements == 1 {
        output[0] = input[0];
        return;
    }

    // Stack of monotone blocks: $(\sum y_i, n_i)$
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(n_elements);

    for &val in input {
        let mut current_sum = val;
        let mut current_count = 1;

        // Merge blocks while monotonicity is violated: $\bar{y}_{prev} \ge \bar{y}_{curr}$
        while let Some(&(prev_sum, prev_count)) = blocks.last() {
            if prev_sum / prev_count as f64 >= current_sum / current_count as f64 {
                current_sum += prev_sum;
                current_count += prev_count;
                blocks.pop();
            } else {
                break;
            }
        }
        blocks.push((current_sum, current_count));
    }

    let mut q_idx = 0;
    for (total_sum, count) in blocks {
        let average_val = total_sum / count as f64;
        for _ in 0..count {
            output[q_idx] = average_val;
            q_idx += 1;
        }
    }
}
