import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# 1)----------------------------------------------------------------------------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_n, y_train)

y_train_knn = knn_model.predict(X_train_n)
y_test_knn = knn_model.predict(X_test_n)

print("KNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn)))

plot_decision_regions(X_train_n, y_train, classifier=knn_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=5), Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
plt.tight_layout()
plt.show()
#dobiveni graf bolje predvida rezultate i to sa 0.825 točnosti logicke regresije na 0.922 sa knn gdje je k=5,
# kod logicke regresije dobili smo pravac dok smo ovdje dobili krivulju i dvije crvene regije u plavome i
# jednu plavu regiju i crvenome dijelu 
# 2)----------------------------------------------------------------------------------------------
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=knn_1)
plt.title("KNN (K=1)")
plt.tight_layout()
plt.show()

y_train_knn1 = knn_1.predict(X_train_n)
y_test_knn1 = knn_1.predict(X_test_n)
print("KNN (K=1): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn1)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn1)))

#sa k=1 dobili smo 7 crvenih regija u plavome dijelu i 5 plavih regija u crvenome,
#to predstavlja overfitting jer je sada tocnost 0.994

knn_100 = KNeighborsClassifier(n_neighbors=100)
knn_100.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=knn_100)
plt.title("KNN (K=100)")
plt.tight_layout()
plt.show()

y_train_knn100 = knn_100.predict(X_train_n)
y_test_knn100 = knn_100.predict(X_test_n)
print("KNN (K=100): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn100)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn100)))

#sa k=100 vise nemamo izmjesanih regija , ali je sada tocnost znacajno manja 
# te to predstavlja underfitting jer je tocnost 0.797
