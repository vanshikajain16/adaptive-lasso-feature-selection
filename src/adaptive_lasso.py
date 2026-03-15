import numpy as np

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

class AdaptiveLasso:

    def __init__(self, lr=0.001, lam=0.1, iterations=2000):
        self.lr = lr
        self.lam = lam
        self.iterations = iterations
        self.loss_history = []

    def fit(self, X, y):

        n, m = X.shape
        self.beta = np.zeros(m)
        self.bias = 0

        for t in range(1, self.iterations + 1):

            y_pred = X @ self.beta + self.bias

            grad_beta = -(1/n) * X.T @ (y - y_pred)
            grad_bias = -(1/n) * np.sum(y - y_pred)

            dynamic_lambda = self.lam / np.sqrt(t)

            temp = self.beta - self.lr * grad_beta

            self.beta = soft_threshold(temp, dynamic_lambda * self.lr)

            self.bias -= self.lr * grad_bias

            loss = (1/(2*n)) * np.sum((y - y_pred)**2)
            self.loss_history.append(loss)

    def predict(self, X):
        return X @ self.beta + self.bias