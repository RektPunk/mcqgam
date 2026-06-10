import numpy as np
from numpy.typing import NDArray

class Regressor:
    """Monotone Quantile Generalized Additive Model."""
    def __init__(
        self, degree: int, num_internal_knots: int, quantiles: list[float]
    ) -> None:
        """Initializes the MQGAM regressor.

        Args:
            degree: Degree of the B-spline basis functions (e.g., 3 for cubic).
            num_internal_knots: Number of internal knots for each feature.
            quantiles: Target quantile levels to fit (must be in [0, 1]).
        """
        ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fits the model to the training data.

        Args:
            x: Input feature matrix of shape (n_samples, n_features).
            y: Target observation vector of shape (n_samples,).
        """
        ...
    def predict(
        self, x: NDArray[np.float64], quantiles: list[float]
    ) -> NDArray[np.float64]:
        """Predicts quantile values for the given features.

        Args:
            x: Input feature matrix of shape (n_samples, n_features).
            quantiles: Quantile levels to predict.

        Returns:
            Matrix of predicted quantile values of shape (n_samples, n_quantiles).
        """
        ...
