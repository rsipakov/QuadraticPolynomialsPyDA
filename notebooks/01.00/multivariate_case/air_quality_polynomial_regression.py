# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Load the Air Quality Dataset
# ------------------------------

# Load the dataset
# Replace 'air_quality.csv-csv_data-combined' with the path to your dataset
data = pd.read_csv('air_quality.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Define the dependent and independent variables
dependent_var = 'HCHO'  # Formaldehyde concentration
independent_vars = ['t_C', 'C6H5OH', 'HCl', 'NH3']  # Temperature, Phenol, Hydrogen Chloride, Ammonia

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
# Prepare Data for Modeling
# ------------------------------

X = data[independent_vars]
y = data[dependent_var]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Fit 3rd Degree Polynomial Regression Model
# ------------------------------

# Generate polynomial features (degree=3)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Fit linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Retrieve the model coefficients
coefficients = model.coef_
intercept = model.intercept_
feature_names = poly.get_feature_names_out(independent_vars)

# Create a DataFrame to display coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print("\nCoefficients of the 3rd-degree polynomial regression model:")
print(coef_df)

# Display the intercept
print(f"\nIntercept of the model: {intercept}")
