# California Housing Prices Prediction

## Project Overview
This project aims to predict the average housing prices in different districts of California using various features such as population density and average income. The model is built using the California Housing Prices Dataset.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Description](#data-description)
4. [Model Evaluation](#model-evaluation)
5. [License](#license)

## Installation
To run this project, you need to have Python installed. You can use the following command to install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Ferhat-Kan/california-housing-prices-prediction.git
cd california-housing-prices-prediction
```
2. Run the Jupyter Notebook:
```bash
jupyter notebook
```

3. Open the notebook file (e.g., housing_prices_prediction.ipynb) and execute the cells to see the results.

## Data Description
The dataset used for this project is the California Housing Prices Dataset, which can be accessed via:
```bash
https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv
```

The dataset contains various features, including:

- `longitude`: Longitude of the district
- `latitude`: Latitude of the district
- `housing_median_age`: Median age of houses
- `total_rooms`: Total number of rooms
- `total_bedrooms`: Total number of bedrooms
- `population`: Total population of the district
- `households`: Total number of households
- `median_income`: Median income of the district
- `median_house_value`: Target variable (median house value)

## Steps to Build the Model

### 1. Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
```

### 2. Load and Explore the Data
```python

# Load the data
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
housing = pd.read_csv(url)

# Display the first 5 rows
print(housing.head())

# General information about the data
print(housing.info())

# Statistical summary
print(housing.describe())
```

### 3. Handle Missing Values
```python

# Check for missing values
print(housing.isnull().sum())

# Fill missing values with the median
housing['total_bedrooms'] = housing['total_bedrooms'].fillna(housing['total_bedrooms'].median())
```

### 4. Encode Categorical Variables
```python
# One-hot encode 'ocean_proximity' categorical variable
housing = pd.get_dummies(housing, columns=['ocean_proximity'])
```

### 5. Split Features and Target Variable
```python
# Target variable (median_house_value)
y = housing['median_house_value']

# Features
X = housing.drop('median_house_value', axis=1)
```

### 6. Split Data into Training and Testing Sets
```python

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7. Scale the Data
```python

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 8. Train the Linear Regression Model
```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

### 9. Evaluate the Model
```python
# Predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
```

### 10. Train a Random Forest Model
```python

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R^2 Score: {r2_rf}")
```

### 11. Hyperparameter Optimization with Grid Search
```python

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
```

### 12. Train the XGBoost Model
```python

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Mean Squared Error: {mse_xgb}")
print(f"XGBoost R^2 Score: {r2_xgb}")
```

### 13. Hyperparameter Optimization for XGBoost
```python

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train_scaled, y_train)

print("Best Parameters for XGBoost:", grid_search_xgb.best_params_)
```

### License
```python

This project is licensed under the MIT License.
```