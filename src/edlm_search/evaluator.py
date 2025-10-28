from typing import Protocol

import numpy as np


class Evaluator(Protocol):
    """A protocol for evaluators that calculate a score based on model predictions."""

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculates and returns a score for a single epoch's predictions."""
        ...
