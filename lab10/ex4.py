import numpy as np

def polynom_roots(poly: np.ndarray) -> np.ndarray:
    n = len(poly) - 1
    C = np.zeros(shape=(n, n))
    C[:, n-1] = -poly[:-1]
    for i in range(1, n):
        C[i, i-1] = 1
    eigen_vals = np.linalg.eigvals(C)
    return eigen_vals

if __name__ == '__main__':
    poly = np.array([-6, 5, 1])
    roots = polynom_roots(poly)
    print('Polynom coefficients:', poly)
    print('Polynom roots:', roots)