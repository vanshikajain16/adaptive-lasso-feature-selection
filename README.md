# Adaptive LASSO Feature Selection

This project implements **Adaptive LASSO** using a custom optimization algorithm
and compares it with standard regression models.

The goal is to demonstrate **feature selection using regularization** and analyze
the trade-off between model accuracy and sparsity.

Dataset used: Boston Housing Dataset.

---

## Models Compared

- Ridge Regression
- LASSO Regression
- Adaptive LASSO (custom implementation)

---

## Results

| Model |Mean Squared Error|
|-------|------------------|
| Ridge Regression | 24.43 |
| LASSO Regression | 25.81 |
| Adaptive LASSO   | 39.39 |

Adaptive LASSO selected **11 out of 13 features**, demonstrating its ability
to reduce dimensionality.

---

