### My solution is wrong

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Parameters
num_trials = 100
train_size = 50
test_size = 10000
dimension = 10

# Means
mu_0 = np.zeros(dimension)
mu_1 = np.array([1,1,1,1,0,0,0,0,0,0])

# Track error rates
rbf_errors = []
linear_errors = []
lda_errors = []

for _ in range(num_trials):
    # Training data
    X_train_0 = np.random.multivariate_normal(mu_0, np.eye(dimension), train_size)
    X_train_1 = np.random.multivariate_normal(mu_1, np.eye(dimension), train_size)
    y_train_0 = np.zeros(train_size)
    y_train_1 = np.ones(train_size)
    
    X_train = np.vstack([X_train_0, X_train_1])
    y_train = np.hstack([y_train_0, y_train_1])
    
    # Shuffle
    X_train, y_train = shuffle(X_train, y_train)
    
    # Test data
    X_test_0 = np.random.multivariate_normal(mu_0, np.eye(dimension), test_size)
    X_test_1 = np.random.multivariate_normal(mu_1, np.eye(dimension), test_size)
    y_test_0 = np.zeros(test_size)
    y_test_1 = np.ones(test_size)
    
    X_test = np.vstack([X_test_0, X_test_1])
    y_test = np.hstack([y_test_0, y_test_1])
    
    # --- SVM RBF ---
    clf_rbf = SVC(C=10)
    clf_rbf.fit(X_train, y_train)
    rbf_pred = clf_rbf.predict(X_test)
    rbf_error = 1 - accuracy_score(y_test, rbf_pred)
    rbf_errors.append(rbf_error)
    
    # --- SVM Linear ---
    clf_linear = SVC(C=10, kernel='linear')
    clf_linear.fit(X_train, y_train)
    linear_pred = clf_linear.predict(X_test)
    linear_error = 1 - accuracy_score(y_test, linear_pred)
    linear_errors.append(linear_error)
    
    # --- LDA ---
    lda = LDA()
    lda.fit(X_train, y_train)
    lda_pred = lda.predict(X_test)
    lda_error = 1 - accuracy_score(y_test, lda_pred)
    lda_errors.append(lda_error)

# Results
print(f"SVM (RBF kernel) average test error: {np.mean(rbf_errors):.4f}")
print(f"SVM (Linear kernel) average test error: {np.mean(linear_errors):.4f}")
print(f"LDA average test error: {np.mean(lda_errors):.4f}")