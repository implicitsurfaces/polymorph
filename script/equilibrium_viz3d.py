import numpy as np
import matplotlib.pyplot as plt
import csv

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

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(param1_values, param2_values, costs, c=costs, cmap='viridis', s=1)
fig.colorbar(scatter, label='Cost')
ax.set_xlabel('Y position')
ax.set_ylabel('Rotation angle')
ax.set_zlabel('Cost')
ax.set_title('3D Scatter Plot of Cost Function')

plt.show()
