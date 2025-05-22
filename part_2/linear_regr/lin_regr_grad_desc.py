import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

class LinearRegressionGradientDescent(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, max_iterations=100, batch_size=32, random_state=None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.theta = None
        self.all_mse = []

    def fit(self, X, y):
        self.all_mse = []
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.c_[np.ones(X.shape[0]), X]  #fill 1st column of X with 1

        #random theta initialization
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.theta = np.random.randn(X.shape[1])

        samples = X.shape[0]

        for epoch in range(self.max_iterations):
            permutation = np.random.permutation(samples)
            X_shuffled = X[permutation]
            y_shuffled = y.iloc[permutation]

            for i in range(0, samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                #gradient = (1/m) * X^T * (X * theta - y)
                m = X_batch.shape[0]
                y_pred = X_batch @ self.theta
                error = y_pred - y_batch
                grad = (1/m) * X_batch.T @ error

                self.theta -= self.learning_rate * grad

            y_pred_full = X @ self.theta
            self.all_mse.append(mse(y, y_pred_full))

        return self

    def predict(self, X):
        if self.theta is None:
            raise ValueError("Predicting cannot be proceed without fitting.")

        if hasattr(X, "toarray"):
            X = X.toarray()

        X_with_1 = np.c_[np.ones(X.shape[0]), X]  #fill 1st column of X with 1

        return X_with_1 @ self.theta

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        square_sum_rest = np.sum((y - y_pred) ** 2)
        square_sum_total = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (square_sum_rest / square_sum_total)
        return r2