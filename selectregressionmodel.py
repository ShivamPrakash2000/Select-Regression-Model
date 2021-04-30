# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def multiple_regression():
    # Training the Multiple Linear Regression model on the Training set
    print("Multiple Regression Model...")
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Evaluating the Model Performance
    return r2_score(y_test, y_pred)

def polynomial_regression():
    # Training the Polynomial Regression model on the Training set
    print("Polynomial Regression Model...")
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(poly_reg.transform(X_test))
    return r2_score(y_test, y_pred)

def support_vector_regression():
    print("Support Vector Regression Model...")

    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train_svr = sc_X.fit_transform(X_train)
    y_train_svr = sc_y.fit_transform(y_train)

    # Training the SVR model on the Training set
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train_svr, y_train_svr)

    # Predicting the Test set results
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))

    # Evaluating the Model Performance
    return r2_score(y_test, y_pred)

def decision_tree_regression():
    print("Decision Tree Regression Model...")

    # Training the Decision Tree Regression model on the Training set
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Evaluating the Model Performance
    return r2_score(y_test, y_pred)

def random_forest_regression():
    print("Random Forest Regression Model...")

    # Training the Random Forest Regression model on the whole dataset
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Evaluating the Model Performance
    return r2_score(y_test, y_pred)

r2 = [multiple_regression(), polynomial_regression(), support_vector_regression(), decision_tree_regression(), random_forest_regression()]
model = ["Multiple", "Polynomial", "Support Vector", "Decision Tree", "Random Forest"]
print("\n*****************************************************************\n")
print("Value of Cofficient of Determination (R^2) for :-\n")
print(f"Multiple Regression Model :- {r2[0]}\n")
print(f"Polynomial Regression Model :- {r2[1]}\n")
print(f"Support Vector Regression Model :- {r2[2]}\n")
print(f"Decision Tree Regression Model :- {r2[3]}\n")
print(f"Random Forest Regression Model :- {r2[4]}\n")
print("\n*****************************************************************\n")

print(f"For this Dataset {model[r2.index(max(r2))]} Regression Model is best, with Value of Cofficient of Determination (R^2) is {max(r2)}\n")
