import numpy as np


class LogME:
    def __init__(self, is_regression: bool=True):
        self.is_regression = is_regression
        self.is_fitted = False
        self.ms = []

    @staticmethod
    def evidence_k(n, D, F, FT, V, VT, y, sigma):
        epsilon = 1e-5
        alpha = 1.
        beta = 1.
        temp_alpha = 1.
        temp_beta = 1.
        alpha_beta = alpha / beta
        temp_m = (VT @ (FT @ y))
        m = np.zeros(D)
        count = 0
        while True:
            count += 1
            if count > 20:
                print("Maximum iteration reached!")
                break
            gamma = (sigma / (sigma + alpha_beta)).sum()
            m = V @ (temp_m / (sigma + alpha_beta))
            temp_alpha = (m * m).sum()
            alpha = gamma / (temp_alpha + epsilon)
            temp_beta = ((F @ m - y) ** 2).sum()
            beta = (n - gamma) / (temp_beta + epsilon)
            new_alpha_beta = alpha / beta
            if np.abs(new_alpha_beta - alpha_beta) < 1e-2:
                break
            alpha_beta = new_alpha_beta
        evidence = (n / 2. * np.log(beta + epsilon) + D / 2. * np.log(alpha + epsilon) - n / 2. * np.log(2 * np.pi)
                    - beta / 2. * (temp_beta + epsilon) - alpha / 2. * (temp_alpha + epsilon)
                    - 0.5 * np.sum(np.log(alpha + beta * sigma + epsilon)))
        return evidence / n, alpha, beta, m

    def fit(self, F: np.ndarray, Y: np.ndarray):
        if self.is_fitted:
            self.__init__(self.is_regression)
        else:
            self.is_fitted = True

        F = F.astype(np.float64)
        if self.is_regression:
            Y = Y.astype(np.float64)
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
        FT = F.T
        n, D = F.shape
        V, sigma, VT = np.linalg.svd(FT @ F)

        evidences = []
        K = Y.shape[1] if self.is_regression else int(Y.max() + 1)
        for i in range(K):
            y = Y[:, i] if self.is_regression else (Y == i).astype(np.float64)
            evidence, _, _, m = self.evidence_k(n, D, F, FT, V, VT, y, sigma)
            evidences.append(evidence)
            self.ms.append(m)
        return np.mean(evidences)

    def predict(self, F: np.ndarray):
        F = F.astype(np.float64)
        one_hot = F @ np.array(self.ms).T
        if self.is_regression:
            return one_hot
        return np.argmax(one_hot, axis=-1)

