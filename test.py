import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Given vertices
v1 = np.array([1, 1])
v2 = np.array([1, 3]) #
v3 = np.array([3, 3])

# Generate points for each segment
n1 = int(np.linalg.norm(v1-v2) / (0.1 * 2))
n2 = int(np.linalg.norm(v2-v3) / (0.1 * 2))

x1 = np.linspace(v1[0], v2[0], n1)
y1 = np.linspace(v1[1], v2[1], n1)

x2 = np.linspace(v2[0], v3[0], n2)[1:]
y2 = np.linspace(v2[1], v3[1], n2)[1:]

print([pair for pair in zip(x1, y1)])
print([pair for pair in zip(x2, y2)])

# Plot
fig, ax = plt.subplots()

# Plot circles for segment 1
for i in range(n1):
  circle = Circle((x1[i], y1[i]), 0.1)
  ax.add_patch(circle)

# Plot circles for segment 2
for i in range(n2-1):
  circle = Circle((x2[i], y2[i]), 0.1)
  ax.add_patch(circle)

ax.set_aspect('equal')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()