# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

# Suppress warnings for clean output
import warnings

warnings.filterwarnings('ignore')

# ------------------------------
# Data Loading and Preparation
# ------------------------------

# Load the dataset
# Replace 'csv_data.csv-csv_data-combined' with your dataset file
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# List of independent variable names and dependent variable name
# Adjust these to match your dataset
independent_vars = ['x1', 'x2']  # For multivariate case
dependent_var = 'y'

# ------------------------------
# Handling Missing Data
# ------------------------------

print("\nHandling missing csv_data...")

# Use KNN Imputer for multivariate imputation
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# Convert the imputed csv_data back to a DataFrame
data = pd.DataFrame(data_imputed, columns=data.columns)

# ------------------------------
# Outlier Detection and Treatment
# ------------------------------

print("\nDetecting outliers...")

# Extract independent variables for outlier detection
X = data[independent_vars]

# Calculate the covariance matrix and its inverse
cov = np.cov(X, rowvar=False)
cov_inv = np.linalg.inv(cov)
mean = X.mean(axis=0).values


# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, mean, cov_inv):
    x_minus_mean = x - mean
    left_term = np.dot(x_minus_mean, cov_inv)
    mahal = np.dot(left_term, x_minus_mean.T)
    return mahal.diagonal()


# Calculate Mahalanobis distance
data['Mahalanobis'] = mahalanobis_distance(X.values, mean, cov_inv)

# Determine the threshold (e.g., p < 0.001)
threshold = chi2.ppf(0.999, df=len(independent_vars))

# Identify outliers
data['Outlier'] = data['Mahalanobis'] > threshold

# Remove outliers
data_cleaned = data[data['Outlier'] == False].reset_index(drop=True)

print(f"Number of outliers removed: {data.shape[0] - data_cleaned.shape[0]}")

# ------------------------------
# Prepare Data for Modeling
# ------------------------------

X = data_cleaned[independent_vars]
y = data_cleaned[dependent_var]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------
# Model Fitting Function
# ------------------------------

def fit_models(X, y, degrees=[1, 2, 3]):
    results = []
    for degree in degrees:
        # Generate polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_poly, y)

        # Predict
        y_pred = model.predict(X_poly)

        # Calculate metrics
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = X_poly.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        # Store results
        results.append({
            'Degree': degree,
            'Model': model,
            'PolynomialFeatures': poly,
            'R-squared': r2,
            'Adjusted R-squared': adj_r2,
            'MSE': mse,
            'RMSE': rmse,
            'Predictions': y_pred
        })
    return results


# ------------------------------
# Fit Models for Single-Variable Case
# ------------------------------

print("\nFitting models for single-variable case...")

# Use only the first independent variable for single-variable case
X_single = X_scaled[:, 0].reshape(-1, 1)

results_single = fit_models(X_single, y)

# ------------------------------
# Fit Models for Multivariate Case
# ------------------------------

print("\nFitting models for multivariate case...")

results_multi = fit_models(X_scaled, y)


# ------------------------------
# Visualization Function
# ------------------------------

def plot_results(X, y, results, title_suffix=''):
    for res in results:
        degree = res['Degree']
        y_pred = res['Predictions']

        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(y)), y, label='Actual', color='blue', alpha=0.6)
        plt.scatter(range(len(y_pred)), y_pred, label=f'Predicted (Degree {degree})', color='red', alpha=0.6)
        plt.title(f'Actual vs Predicted Values {title_suffix} (Degree {degree})')
        plt.xlabel('Sample Index')
        plt.ylabel(dependent_var)
        plt.legend()
        plt.show()

        # Print metrics
        print(f"Degree {degree} {title_suffix} Model Metrics:")
        print(f"R-squared: {res['R-squared']:.4f}")
        print(f"Adjusted R-squared: {res['Adjusted R-squared']:.4f}")
        print(f"MSE: {res['MSE']:.4f}")
        print(f"RMSE: {res['RMSE']:.4f}")
        print("-" * 40)


# ------------------------------
# Plot Results for Single-Variable Case
# ------------------------------

print("\nResults for single-variable models:")

plot_results(X_single, y, results_single, title_suffix='(Single Variable)')

# ------------------------------
# Plot Results for Multivariate Case
# ------------------------------

print("\nResults for multivariate models:")

plot_results(X_scaled, y, results_multi, title_suffix='(Multivariate)')
