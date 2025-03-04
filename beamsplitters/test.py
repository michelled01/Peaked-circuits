import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Create a figure
fig, ax = plt.subplots()

# Example plot lines (just for context)
ax.plot([0, 1], [0, 1], color="blue", label="Line 1")
ax.plot([0, 1], [1, 0], color="red", label="Line 2")

# Custom legend entry with two colors
custom_line = [
    Line2D([0], [0], color="blue", lw=2),
    Line2D([0], [0], color="red", lw=2)
]

ax.legend(custom_line, ["Mixed Color Line"])

plt.show()