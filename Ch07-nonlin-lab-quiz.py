import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Load the CSV into a DataFrame
df = pd.read_csv('7.Py.1.csv')

# Step 2: Plot y vs x
df.plot(x='x', y='y', kind='scatter')
plt.title('y vs x')
plt.show()

# Step 3: Linear Regression (y on x)
X_linear = df[['x']]  # independent variable
y = df['y']           # dependent variable

lin_reg = LinearRegression()
lin_reg.fit(X_linear, y)

print(f"Linear Regression Slope Coefficient: {lin_reg.coef_[0]}")

# Step 4: Add quadratic term (x^2)
df['x_squared'] = df['x'] ** 2
X_quad = df[['x', 'x_squared']]

quad_reg = LinearRegression()
quad_reg.fit(X_quad, y)

print(f"Quadratic Model Coefficient for x: {quad_reg.coef_[0]}")