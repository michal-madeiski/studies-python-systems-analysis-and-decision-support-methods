import numpy as np
class LinearRegressionClosedFormula:
    def __init__(self):
        self._a_ = None  #coefficients of x
        self._b_ = None  #bias

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X_with_1 = np.c_[np.ones(X.shape[0]), X] #fill 1st column of X with 1

        #theta = (X^T X)^-1 X^T y
        XT = X_with_1.T
        XTX = np.dot(XT, X_with_1)
        XTX_inv = np.linalg.pinv(XTX)
        XTy = np.dot(XT, y)
        theta = np.dot(XTX_inv, XTy)

        self._b_ = theta[0]
        self._a_ = theta[1:]

        return self

    def predict(self, X):
        if self._a_ is None or self._b_ is None:
            raise ValueError("Predicting cannot be proceed without fitting.")

        if hasattr(X, "toarray"):
            X = X.toarray()

        return np.dot(X, self._a_) + self._b_

    def score(self, X, y):
        y_pred = self.predict(X)
        square_sum_rest = np.sum((y - y_pred) ** 2)
        square_sum_total = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (square_sum_rest / square_sum_total)
        return r2