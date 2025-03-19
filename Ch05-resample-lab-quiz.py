import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Step 1: Load CSV file
df = pd.read_csv('5.Py.1.csv')

# Step 2: Define predictors and response
X = df[['X1', 'X2']]
X = sm.add_constant(X)  # Adds intercept term
y = df['y']

# Step 3: Fit linear regression model
model = sm.OLS(y, X).fit()

# Step 4: Get standard errors
print(model.summary())

# Specifically extract standard error for β1 (X1)
se_beta1 = model.bse['X1']
print(f"Standard Error for β1 (X1): {se_beta1}")

# # Plot all columns
# df.plot()

# plt.title('Plot of y, X1, and X2')
# plt.show()

# Setup
n_bootstraps = 1000
beta1_estimates = []

# Bootstrap loop
for _ in range(n_bootstraps):
    # Sample with replacement
    sample = df.sample(n=len(df), replace=True)
    
    # Prepare X and y
    X_sample = sample[['X1', 'X2']]
    X_sample = sm.add_constant(X_sample)
    y_sample = sample['y']
    
    # Fit model
    model = sm.OLS(y_sample, X_sample).fit()
    
    # Store beta1 estimate
    beta1_estimates.append(model.params['X1'])

# Calculate standard error
se_beta1_bootstrap = np.std(beta1_estimates, ddof=1)
print(f"Bootstrap s.e.(β1): {se_beta1_bootstrap:.4f}")

n_bootstraps = 1000
block_size = 100
num_blocks = 10
n_rows = len(df)
beta1_estimates = []

# Compute total number of blocks
total_blocks = n_rows // block_size

# Bootstrap loop
for _ in range(n_bootstraps):
    blocks = []
    for _ in range(num_blocks):
        # Random block index
        block_idx = np.random.randint(0, total_blocks)
        # Add block slice
        blocks.append(slice(block_idx * block_size, (block_idx + 1) * block_size))
    
    # Combine blocks
    new_rows = np.r_[tuple(blocks)]
    new_df = df.iloc[new_rows]
    
    # Fit model
    X_sample = new_df[['X1', 'X2']]
    X_sample = sm.add_constant(X_sample)
    y_sample = new_df['y']
    
    model = sm.OLS(y_sample, X_sample).fit()
    beta1_estimates.append(model.params['X1'])

# Calculate standard error
se_beta1_block_bootstrap = np.std(beta1_estimates, ddof=1)
print(f"Block Bootstrap s.e.(β1): {se_beta1_block_bootstrap:.4f}")