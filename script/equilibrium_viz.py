import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

three_d = "--3d" in sys.argv

if len(sys.argv) < 2 or not os.path.exists(sys.argv[-1]):
    print(f"usage: python {sys.argv[0]} [--3d] <filename>")
    sys.exit(1)

filename = sys.argv[-1]

# Read the cost function data from a CSV file
parameters = []
costs = []
with open(filename) as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row if present
    for row in csv_reader:
        param1, param2, cost = map(float, row)
        parameters.append([param1, param2])
        costs.append(cost)

# Convert the data to numpy arrays
parameters = np.array(parameters)
costs = np.array(costs)

# Extract the parameter values
param1_values = parameters[:, 0]
param2_values = parameters[:, 1]

# Create a 3D scatter plot
if three_d:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        param1_values, param2_values, costs, c=costs, cmap="viridis", s=1
    )
    ax.set_zlabel("Cost")
else:
    fig, ax = plt.subplots()
    scatter = ax.scatter(param1_values, param2_values, c=costs, cmap="viridis", s=1)
fig.colorbar(scatter, label="Cost")
ax.set_xlabel("Y position")
ax.set_ylabel("Rotation angle")

plt.show()
