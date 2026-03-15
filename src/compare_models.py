from preprocess import load_data
from adaptive_lasso import AdaptiveLasso
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# load dataset
X_train, X_test, y_train, y_test = load_data()

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# LASSO Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

# Adaptive LASSO (your algorithm)
adaptive = AdaptiveLasso(lr=0.001, lam=8, iterations=2000)
adaptive.fit(X_train, y_train)
adaptive_pred = adaptive.predict(X_test)
adaptive_mse = mean_squared_error(y_test, adaptive_pred)

# print results
print("Model Performance Comparison")
print("----------------------------")
print("Ridge MSE:", ridge_mse)
print("LASSO MSE:", lasso_mse)
print("Adaptive LASSO MSE:", adaptive_mse)

# feature selection info
print("\nAdaptive LASSO selected features:", (adaptive.beta != 0).sum())

print("\nNumber of features selected by Adaptive LASSO:", (adaptive.beta != 0).sum())