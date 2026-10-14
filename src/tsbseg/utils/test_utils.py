import numpy as np


def mock_loader():
    """
    Returns a list containing a single synthetic trial.
    X: (100, 2) random float
    y: (100,) int, with a change point at 50
    """
    rng = np.random.default_rng(42)
    X = rng.random((100, 2))
    y = np.zeros(100, dtype=int)
    y[50:] = 1
    return [X], [y]
