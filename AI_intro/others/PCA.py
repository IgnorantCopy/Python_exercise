import numpy as np

X = np.array([
    [-1., 0.5, 1., -0.5],
    [-1., -0.5, 1., 0.5],
])
target = X @ X.T
print(target)
result = np.linalg.eig(target)
print(result[0])
print(result[1])

W = result[1].T
print(W)
sigma_square = 0
for i in range(4):
    temp = W.T @ X[:, i].T
    sigma_square += (temp ** 2).sum()
print(sigma_square)
