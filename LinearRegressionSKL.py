from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
data = pd.read_csv("FuelConsumption.csv")

# Extract features (X) and labels (Y)
X = data[['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION']].to_numpy()
Y = data['COEMISSIONS '].to_numpy()

# Split data into training (80%), and test (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# set up the scaler for the features in the training set
x_train_scaler = StandardScaler().fit(X_train)

# scale the features in the training set
X_train_scaled = x_train_scaler.transform(X_train)
# scale the features in the test set
X_test_scaled = x_train_scaler.transform(X_test)

# instantiate the regression module
regressor = LinearRegression()

# create the model
model = regressor.fit(X_train_scaled, Y_train)

# get the predictions on the test set
Y_test_pred = model.predict(X_test_scaled)

# The coefficients
print('Coefficients: \n', model.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_test_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, Y_test_pred))