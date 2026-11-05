import numpy as np

from .base import BaseLinearClassifier

class AdalineGD(BaseLinearClassifier):
    """ADAptative LInear NEuron Classifier (Gradient Descent).

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function values in each epoch.
    
    """
    
    def fit(self, X, y):     
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.

        y : array-like, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : object

        """
        self.initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

class AdalineSGD(BaseLinearClassifier):
    """ADAptative LInear NEuron Classifier. (Stochastic Gradient Descent)

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all
        training examples in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None,
                 w_initialized=False):
        """
        Initialize attributes of the parent class.
        Then initialize attributes specific to AdalineSGD.
        """
        super().__init__(eta=eta, n_iter=n_iter, random_state=random_state, 
                         w_initialized=w_initialized)
        self.shuffle = shuffle

    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.

        y : array-like, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : object

        """
        self.initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = [self._update_weights(xi, target) for xi, target in zip(X, y)]
            self.losses_.append(np.mean(losses))
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights."""
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights."""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2 * xi * error
        self.b_ += self.eta * 2 * error
        loss = error**2
        return loss
