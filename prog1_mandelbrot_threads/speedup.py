import matplotlib.pyplot as plt
import numpy as np

# Given data
threads = np.array([2, 3, 4, 5, 6, 7, 8])
speedup = np.array([1.96, 1.66, 2.43, 2.48, 3.28, 3.35, 3.97])

# Ideal linear speedup for reference
ideal_speedup = threads


plt.figure(figsize=(8, 5))
plt.plot(threads, speedup, marker='o', linestyle='-', label="Measured Speedup", color='b')
plt.plot(threads, ideal_speedup, linestyle='--', label="Ideal Linear Speedup", color='r')

plt.xlabel("Number of Threads")
plt.ylabel("Speedup (Relative to Serial)")
plt.title("Speedup vs. Number of Threads")
plt.legend()
plt.grid(True)


plt.show()
