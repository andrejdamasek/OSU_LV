from sklearn import datasets
from sklearn.model_selection import train_test_split
# ucitaj ugradeni podatkovni skup
X, y = datasets.load_diabetes(return_X_y=True)
# podijeli skup na podatkovni skup za ucenje i podatkovni skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)


from sklearn.preprocessing import MinMaxScaler
# min-max skaliranje
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
#X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()


#class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

import sklearn.linear_model as lm
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

from sklearn.metrics import mean_absolute_error
#predikcija izlazne velicine na skupu podataka za testiranje
y_test_p = linearModel.predict(X_test_n)
#evaluacija modela na skupu podataka za testiranje pomocu MAE
MAE = mean_absolute_error(y_test, y_test_p)

