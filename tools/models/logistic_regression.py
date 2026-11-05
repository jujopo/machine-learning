import numpy as np

from .base import BaseLinearClassifier

class LogisticRegressionGD(BaseLinearClassifier):
    """Gradient descent-based logistic regression classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
        
    Atrtributes
    -------------
    w_ : 1d-array
        Weights after training.
    b_: Scalar
        Bias unit after fitting. 
    losses_ : list
        Mean squared error loss function values in each epoch.

    """

    def fit(self, X, y):
        """Fit training data
        
        Parameters
        ------------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.
        y: array-like, shape = [n_examples]
            Target values.

        Returns
        ----------
        self : Instance of LogisticRegressionGD

        """
        self.initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2 * errors.mean()
            loss = (-y.dot(np.log(output)) - 
                    ((1 - y).dot(np.log(1 - output)))
                    / X.shape[0])
            self.losses_.append(loss)
        return self

    def activation(self, z):
        """Compute logistic sigmoid activation."""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))