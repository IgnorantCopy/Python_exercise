from numpy.ma.core import argmin

centers = [(6.2, 3.2), (6.6, 3.7), (6.5, 3.0)]
xs = [
    (5.9, 3.2),
    (4.6, 2.9),
    (6.2, 2.8),
    (4.7, 3.2),
    (5.5, 4.2),
    (5.0, 3.0),
    (4.9, 3.1),
    (6.7, 3.1),
    (5.1, 3.8),
    (6.0, 3.0),
]

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

for epoch in range(3):
    print(f"Epoch {epoch+1}")
    clusters = [[] for _ in range(3)]
    for i, x in enumerate(xs):
        print(f"x{i+1}:")
        d1 = dist(x, centers[0])
        d2 = dist(x, centers[1])
        d3 = dist(x, centers[2])
        print(d1, d2, d3)
        index = argmin([d1, d2, d3])
        print(f"x{i+1} belongs to cluster {index+1}")
        clusters[index].append(i+1)
    print(f"Clusters: {clusters}")
    for i, c in enumerate(clusters):
        sum1 = 0
        sum2 = 0
        for j in c:
            sum1 += xs[j - 1][0]
            sum2 += xs[j - 1][1]
        centers[i] = (sum1 / len(c), sum2 / len(c))
        print(f"New center{i+1}: {centers[i]}")
    print("-" * 50)
