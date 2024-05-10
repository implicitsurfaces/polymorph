import numpy as np
import matplotlib.pyplot as plt
import csv
import mplcursors

# Read the cost function data from a CSV file
parameters = []
costs = []
with open('cost_function_data.csv', 'r') as file:
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

# Create a scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(param1_values, param2_values, c=costs, cmap='viridis', s=10)
fig.colorbar(scatter, label='Cost')
ax.set_xlabel('Y position')
ax.set_ylabel('Rotation angle')
ax.set_title('Scatter Plot of Cost Function')

# Enable hover annotations
cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"Cost: {sel.target[2]:.2f}"))

plt.show()
