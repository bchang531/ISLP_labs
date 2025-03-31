#!/usr/bin/env python
"""
This script loads the dataset from "10.R.RData" containing:
  - x (training features, n x p with n=300, p=200)
  - y (training response)
  - x.test (test features)
  - y.test (test response)

It then:
  1. Concatenates x and x.test (vertically) and standardizes each column.
  2. Performs PCA on the concatenated, scaled data.
  3. Computes the proportion of total variance explained by the first 5 PCs.
  4. Uses the first 5 PC scores as predictors to fit a linear regression model on the training data
     and computes the test MSE.
  5. Fits an ordinary least squares regression using the original features and computes its test MSE.
"""

import numpy as np
import pyreadr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # Load the RData file
    result = pyreadr.read_r("10.R.RData")
    # The file should contain: 'x', 'y', 'x.test', and 'y.test'
    x = result['x']
    y = result['y']
    x_test = result['x.test']
    y_test = result['y.test']
    
    print("Training features shape:", x.shape)
    print("Test features shape:", x_test.shape)
    
    # Concatenate x and x_test (similar to rbind in R)
    X_all = np.vstack([x, x_test])
    
    # Standardize the features (center and scale to unit variance)
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    # Perform PCA on the concatenated scaled data
    pca = PCA()
    pca.fit(X_all_scaled)
    
    # Proportion of variance explained by the first five principal components
    var_explained_5 = pca.explained_variance_ratio_[:5].sum()
    print(f"Proportion of variance explained by the first 5 PCs: {var_explained_5:.4f}")
    
    # Transform the data to obtain PC scores
    X_all_pcs = pca.transform(X_all_scaled)
    n_train = x.shape[0]
    
    # Split the PC scores back into training and test sets, and take only the first 5 PCs
    X_train_pcs = X_all_pcs[:n_train, :5]
    X_test_pcs = X_all_pcs[n_train:, :5]
    
    # Fit a linear regression model using the first 5 principal components
    lr_pcs = LinearRegression()
    lr_pcs.fit(X_train_pcs, y)
    y_pred_pcs = lr_pcs.predict(X_test_pcs)
    mse_pcs = mean_squared_error(y_test, y_pred_pcs)
    print(f"MSE using first 5 PCs: {mse_pcs:.4f}")
    
    # Fit an OLS linear regression using the original features
    lr_orig = LinearRegression()
    lr_orig.fit(x, y)
    y_pred_orig = lr_orig.predict(x_test)
    mse_orig = mean_squared_error(y_test, y_pred_orig)
    print(f"MSE using original features: {mse_orig:.4f}")

if __name__ == '__main__':
    main()