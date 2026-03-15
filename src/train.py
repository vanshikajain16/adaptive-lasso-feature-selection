from preprocess import load_data
from adaptive_lasso import AdaptiveLasso
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = load_data()

model = AdaptiveLasso(lr=0.001, lam=0.5, iterations=2000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)

print("Adaptive Lasso MSE:", mse)