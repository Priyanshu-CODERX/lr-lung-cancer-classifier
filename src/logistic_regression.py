import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, iterations=1000):
        """
        Initialize the Logistic Regression Model

        Parameters
        ----------
        lr : float
            Learning rate for gradient descent
        iterations : int
            Number of iterations to run gradient descent
        """
        self.lr = lr
        self.iterations = iterations
        self.W = None  # Model weights (coefficients)
        self.b = None  # Bias (intercept)
        self.loss_history = []  # Stores cost function value per iteration

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        """
        z = np.clip(z, -500, 500)  # prevent overflow
        return 1.0 / (1.0 + np.exp(-z))

    def compute_cost(self, X, y, w, b):
        """
        Compute the logistic regression cost
        """
        m = X.shape[0]
        z = np.dot(X, w) + b
        f_wb = self.sigmoid(z)
        cost = -(1 / m) * np.sum(
            y * np.log(f_wb + 1e-15) + (1 - y) * np.log(1 - f_wb + 1e-15)
        )
        return cost

    def compute_gradient(self, X, y, w, b):
        """
        Compute gradient for logistic regression cost function
        """
        m = X.shape[0]
        z = np.dot(X, w) + b
        f_wb = self.sigmoid(z)
        error = f_wb - y

        dj_dw = (1 / m) * np.dot(X.T, error)
        dj_db = (1 / m) * np.sum(error)

        return dj_dw, dj_db

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        """
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, y, self.W, self.b)
            self.W -= self.lr * dj_dw
            self.b -= self.lr * dj_db

            cost = self.compute_cost(X, y, self.W, self.b)
            self.loss_history.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i:4d}: Cost {cost:.4f}")

    def predict_proba(self, X):
        """
        Predict probability estimates for input samples
        """
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels (0 or 1)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        """
        Compute accuracy of the model
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
