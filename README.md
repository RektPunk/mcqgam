<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=mcqgam&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
</div>


**mcqgam** is a library for fitting a **Monotone Composite Quantile Generalized Additive Model** using B-splines and ADMM optimization. 
The model estimates a conditional quantile function by minimizing the empirical Continuous Ranked Probability Score (CRPS), approximated via weighted Huber-smoothed pinball losses across quantile levels under the non-crossing constraint.

## Installation
```bash
pip install mcqgam
```

## Key Features
- Optimized B-spline construction using De Boor's recursion.
- Built-in structural monotonicity via coefficient constraints.
- Parallelized ADMM solver with fixed-Hessian Newton updates and PAVA.
- Monotone Cubic Spline interpolation for off-grid quantile predictions.
