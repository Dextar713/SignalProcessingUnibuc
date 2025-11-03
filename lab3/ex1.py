import numpy as np
import matplotlib.pyplot as plt

N = 8
F = np.zeros(shape=(N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        F[k][n] = np.exp(-2j * np.pi * k * n / N)

# F_unitary = F / np.sqrt(N)
print(np.allclose(np.round(F.conj().T@F, 3)/N, np.eye(N)))

fig, axs = plt.subplots(nrows=N, ncols=2, figsize=(10, 12))
for k in range(N):
    axs[k, 0].plot(np.real(F[k, :]))
    axs[k, 0].set_title(f"Row {k} - Real Part")
    axs[k, 1].plot(np.imag(F[k, :]))
    axs[k, 1].set_title(f"Row {k} - Imag Part")
plt.tight_layout()
plt.savefig("ex1.pdf")
plt.show()
