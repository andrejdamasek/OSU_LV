from sklearn.model_selection import train_test_split
import pandas as pd
import math
import numpy as np
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, max_error


data = pd.read_csv('data_C02_emission.csv')

input = ['Engine Size (L)', 'Fuel Type', 'Cylinders', 'Fuel Consumption City (L/100km)', 
                  'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']

output = 'CO2 Emissions (g/km)'

X = data[input].copy()
y = data[output]

ohe = OneHotEncoder(sparse_output=False) 
encoded = ohe.fit_transform(X[['Fuel Type']])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['Fuel Type']))

X_encoded = pd.concat([X, encoded_df], axis=1)
X_encoded = X_encoded.drop(columns=['Fuel Type'])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)


linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_p = linearModel.predict(X_test)


MSE = mean_squared_error(y_test, y_test_p)
RMSE = math.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2_SCORE = r2_score(y_test, y_test_p)

print(f"MSE: {MSE}, RMSE: {RMSE}, MAE: {MAE}, MAPE: {MAPE}, R2 SCORE: {R2_SCORE}")

ME = max_error(y_test, y_test_p)
print(f"Max Error: {ME}")

errors = np.abs(y_test - y_test_p)
max_error_id = np.argmax(errors)
vehicle_model = data.iloc[X_test.index[max_error_id]]['Model'] 
print(f"Model with max error: {vehicle_model}")
