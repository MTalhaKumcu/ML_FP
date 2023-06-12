# MEHMET_TALHA_KUMCU_38306
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("final_test_regression_data.csv")
# PLS ENTER YOUR DATA PATH
# I work same folder cuz of I did not change my path

# Split the data into features (X) and target variable (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create holdout datasets (train: 60%, test: 20%, validation: 20%)
# YOU CAN CHANGE YOUR test_size=, random_state and your dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Here
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42  # Here
)

regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)

regressor2 = LinearRegression()
regressor2.fit(X_train, y_train)

y_train_pred = regressor1.predict(X_train)
y_test_pred = regressor1.predict(X_test)
y_val_pred = regressor1.predict(X_val)
# YOU CAN CHANGE YOUR colours
plt.scatter(X_train, y_train, color="blue", label="Train Data")  # Here
plt.plot(X_train, y_train_pred, color="pink", label="Train Predictions")  # Here
plt.plot(X_test, y_test_pred, color="green", label="Test Predictions")  # Here
plt.plot(X_val, y_val_pred, color="turquoise", label="Validation Predictions")  # Here
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regression Model Predictions")
plt.legend()
plt.show()

# MEHMET_TALHA_KUMCU_38306
