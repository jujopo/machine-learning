import numpy as np

from .base import BaseLinearClassifier

class Perceptron(BaseLinearClassifier):
    """Perceptron classifier.
    
    Parameters
    ------------
    eta: float
      Learning rate (between 0.0 and 1.0)
    n_iter: int
      Passes over the training dataset.
    random_state: int
      Random number generator seed for random weight initialization. 
    
    Attributes
    ------------
    w_: 1d-array
      Weights after fitting.
    b_: float
      Bias unit after fitting.

    errors_: list
      Number of missclassifications (updates) in each epoch.
    """

    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        -------------
        X: {array-like}, shape = [n_exampes, n_features]
          Training vectors, where n_examples is the number of examples
          and n_features is the number of features.
        y: array-like, shape = [n_examples]
          Target values.
        
        Returns
        -------------
        self: object
        """
        self.initialize_weights(X.shape[1])  # Create self.w_ and self.b_
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        """Return class labels after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)