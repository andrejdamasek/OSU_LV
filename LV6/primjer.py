from os import pipe
from sklearn . svm import SVC
from sklearn . preprocessing import StandardScaler
from sklearn . pipeline import Pipeline
from sklearn . pipeline import make_pipeline
from sklearn . model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

#data.hist()
#plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_n,y_train)
param_grid = { 'model__C': [ 10 , 100 , 100 ] , 
                'model__gamma': [ 10, 1, 0.1, 0.01 ] }
svm_gscv = GridSearchCV ( knn , param_grid , cv =5 , scoring = 'accuracy' ,n_jobs = - 1 )
svm_gscv.fit ( X_train_n , y_train )
print(svm_gscv.best_params_)
print(svm_gscv.best_score_)
print(svm_gscv.cv_results_)