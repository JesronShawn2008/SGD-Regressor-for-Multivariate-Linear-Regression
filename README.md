# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a dataset using a dictionary, convert it into a DataFrame, and apply one-hot encoding to convert the categorical city column into numerical form.

2. Split the data into features (X) and two target variables (Price and Occupants), then divide the dataset into training and testing sets.

3. Standardize the feature values using a scaling technique to ensure proper convergence for SGD-based training.

4. Train two separate SGDRegressor models (one for Price and one for Occupants), predict values on the test set, and display both predicted and actual outputs

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = {
    "Square_Feet": [1200, 1500, 1800, 2500, 900, 2000, 1750, 3000],
    "Bedrooms":     [2, 3, 3, 4, 2, 3, 3, 5],
    "Age":          [10, 5, 12, 3, 20, 8, 6, 2],
    "City":         ["Chennai", "Chennai", "Mumbai", "Delhi", "Delhi", "Mumbai", "Chennai", "Delhi"],
    "Price":        [50, 70, 90, 130, 40, 100, 85, 150],    # Lakhs
    "Occupants":    [3, 4, 4, 5, 3, 5, 4, 6]
}

df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=["City"], drop_first=True)

X = df.drop(["Price", "Occupants"], axis=1)
y = df[["Price", "Occupants"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = SGDRegressor(max_iter=2000, tol=0.01, learning_rate="constant")

price_model = SGDRegressor(max_iter=2000, tol=0.01)
price_model.fit(X_train_scaled, y_train["Price"])

occupants_model = SGDRegressor(max_iter=2000, tol=0.01)
occupants_model.fit(X_train_scaled, y_train["Occupants"])
pred_price = price_model.predict(X_test_scaled)
pred_occupants = occupants_model.predict(X_test_scaled)
print("Predicted Price:", pred_price)
print("Actual Price:", list(y_test["Price"]))
print("Predicted Occupants:", pred_occupants)
print("Actual Occupants:", list(y_test["Occupants"]))

Developed by: Jesron Shawn C J 
RegisterNumber:  25012933
*/
```

## Output:
<img width="1131" height="105" alt="image" src="https://github.com/user-attachments/assets/d240ecdc-8293-4c50-aa27-54b310253585" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
