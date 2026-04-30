import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(0)

# Number of samples
n_samples = 100

# Generate independent variables
x1 = np.random.uniform(0, 10, n_samples)
x2 = np.random.uniform(0, 10, n_samples)

# Generate noise
noise = np.random.normal(0, 5, n_samples)

# Define the true relationship (including linear, quadratic, and interaction terms)
y = (3 * x1) + (2 * x2) + (0.5 * x1**2) - (0.3 * x2**2) + (1.5 * x1 * x2) + noise

# Create a DataFrame
data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

# Save the dataset to a CSV file
data.to_csv('csv_data.csv-csv_data-combined', index=False)

print("Dataset 'csv_data.csv-csv_data-combined' has been created.")
