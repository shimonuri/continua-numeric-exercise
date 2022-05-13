import matplotlib.pyplot as plt
import numpy as np


def numeric_better(x, h):
    return (np.cos(x + h) - np.cos(x - h)) / (2 * h)


def numeric_worse(x, h):
    return (np.cos(x + h) - np.cos(x)) / h


# line 1 points
x = 1
distances = [10 ** (-i) for i in range(1, 16)]
real_y = -np.sin(x)
numeric_better_y = [abs(numeric_better(x, distance) - real_y) for distance in distances]
numeric_worse_y = [abs(numeric_worse(x, distance) - real_y) for distance in distances]

plt.plot(distances, numeric_worse_y, label="real derivative")
plt.plot(distances, numeric_better_y, label="numeric derivative")

# plt.loglog(x, real_y, label="real derivative")

# # plotting the line 1 points
# line 2 points
# plotting the line 2 points
plt.xlabel("h")
# Set the y axis label of the current axis.
plt.ylabel("f`(x)")
# Set a title of the current axes.
plt.title("Two or more lines on same plot with suitable legends ")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
