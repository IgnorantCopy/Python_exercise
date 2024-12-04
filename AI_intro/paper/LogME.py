import numpy as np


class LogME:
    def __init__(self, is_regression: bool=True):
        self.is_regression = is_regression
        self.is_fitted = False
        self.alphas = []
        self.betas = []
        self.ms = []

    @staticmethod
    def evidence_k(n, D, F, FT, V, VT, y, sigma):
        epsilon = 1e-5
        alpha = 1.
        beta = 1.
        temp_m = VT @ (FT @ y)
        temp_alpha = 0.
        temp_beta = 0.
        m = np.zeros(D)
        count = 0
        while True:
            count += 1
            if count > 20:
                print("Maximum iteration reached!")
                break
            gamma = (beta * sigma / alpha + beta * sigma).sum()
            lambda_inverse = np.diag(1. / alpha + beta * sigma)
            m = beta * (V @ (lambda_inverse @ temp_m))
            temp_alpha = (m * m).sum()
            new_alpha = gamma / (temp_alpha + epsilon)
            temp_beta = ((F @ m - y) ** 2).sum()
            new_beta = (n - gamma) / (temp_beta + epsilon)
            if np.abs(new_alpha - alpha) < 1e-5 and np.abs(new_beta - beta) < 1e-5:
                alpha = new_alpha
                beta = new_beta
                break
            alpha = new_alpha
            beta = new_beta
        evidence = n / 2. * np.log(beta) + D / 2. * np.log(alpha) - n / 2. * np.log(2 * np.pi) - beta / 2. * (temp_beta + epsilon) - alpha / 2. * (temp_alpha + epsilon) - 0.5 * np.sum(np.log(alpha + beta * sigma))
        return evidence / n, alpha, beta, m

    def fit(self, F: np.ndarray, Y: np.ndarray):
        if self.is_fitted:
            self.__init__(self.is_regression)
        self.is_fitted = True

        F = F.astype(np.float64)
        Y = Y.astype(np.float64)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        FT = F.T
        n, D = F.shape
        V, sigma, VT = np.linalg.svd(FT @ F)

        evidences = []
        K = Y.shape[1]
        for i in range(K):
            y = Y[:, i]
            evidence, alpha, beta, m = self.evidence_k(n, D, F, FT, V, VT, y, sigma)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        return np.mean(evidences)

    def predict(self, F: np.ndarray):
        F = F.astype(np.float64)
        one_hot = F @ np.array(self.ms).T
        if self.is_regression:
            return one_hot
        return np.argmax(one_hot, axis=-1)

