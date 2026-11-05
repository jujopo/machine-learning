import numpy as np


class BaseLinearClassifier:
    """Shared structure for single-layer linear classifiers."""

    def __init__(self, eta=0.01, n_iter=50, random_state=1, w_initialized=False):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_initialized = w_initialized
        self.rgen = np.random.RandomState(random_state)

    def initialize_weights(self, n_features):
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = np.float64(0.0)
        self.w_initialized = True

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute activation."""
        return z    # Identity activation by default. Override in subclasses.

    def predict(self, X):
        """Return class label after unit step."""
        # Set prediction decision by default as if using sigmoid activation.
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit()")