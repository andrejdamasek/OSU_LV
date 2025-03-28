from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

data = pd.read_csv ('data_C02_emission.csv')
#a)
input=['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']

output='CO2 Emissions (g/km)'

X=data[input]
y=data[output]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

#b)
plt.scatter(X_train['Fuel Consumption Comb (L/100km)'],y_train, c='Blue')
plt.scatter(X_test['Fuel Consumption Comb (L/100km)'],y_test, c='Red')
plt.xlabel('Fuel Consumption Comb (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

#c)
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)

plt.subplot(1, 2, 1)
plt.plot()
plt.hist(X_train['Fuel Consumption Comb (L/100km)'])
plt.subplot(1, 2, 2)
plt.hist(X_train_n[:,5])
plt.show()

X_test_n = sc.transform(X_test)

#d)
linearModel=lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print("Coefficients: ",linearModel.coef_)
#e)
y_test_p = linearModel.predict(X_test_n)

plt.scatter(y_test,y_test_p)
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.show()

#f)
MSE=mean_squared_error(y_test, y_test_p)
RMSE=math.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE=mean_absolute_percentage_error(y_test, y_test_p)
R_TWO_SCORE = r2_score(y_test, y_test_p)

print(f"MSE: {MSE}, RMSE: {RMSE},MAE: {MAE}, MAPE: {MAPE}, R2 SCORE: {R_TWO_SCORE}")

#g)
#ako za treniranje uzmemo 60%,a za test 40% 
#MSE: 230.59375787723604, RMSE: 15.18531388800495,MAE: 8.019191851165152, MAPE: 0.029472917634791145, R2 SCORE: 0.9391456928098105
#ako za treniranje uzmemo 80%,a za test 20% 
#MSE: 257.2002271349325, RMSE: 16.03746323877104,MAE: 8.18449505021558, MAPE: 0.02989202948172508, R2 SCORE: 0.9364108013367262
#ako za treniranje uzmemo 90%,a za test 10% 
#MSE: 222.9096235288416, RMSE: 14.930158188339519,MAE: 8.374511932606104, MAPE: 0.03125276808087804, R2 SCORE: 0.9490770415447916