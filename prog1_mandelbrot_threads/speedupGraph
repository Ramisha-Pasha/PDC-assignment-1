#graph
import matplotlib.pyplot as plt

# Data from your table
threads = [2, 3, 4, 5, 6, 7, 8]
execution_time = [302.044, 360.108, 300.8, 304.399, 304.434, 313.691, 311.45]
speedup = [2.01, 1.65, 1.98, 1.94, 1.94, 1.89, 1.93]
ideal_speedup = [t for t in threads]  # Ideal linear speedup

# Plot the speedup graph
plt.figure(figsize=(8, 6))
plt.plot(threads, speedup, marker='o', linestyle='-', color='b', label="Observed Speedup")
plt.plot(threads, ideal_speedup, linestyle='--', color='r', label="Ideal Speedup (Linear)")

plt.xlabel("Number of Threads")
plt.ylabel("Speedup Factor")
plt.title("Speedup vs. Number of Threads (View 1)")
plt.legend()
plt.grid()
plt.show()
