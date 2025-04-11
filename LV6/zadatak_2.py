import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl)


data = pd.read_csv("Social_Network_Ads.csv")
X = data[["Age", "EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=10)

param_grid = {'n_neighbors': list(range(1, 51))}
knn=KNeighborsClassifier()

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_n, y_train)

best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_
best_knn_model = grid_search.best_estimator_

print("Optimalna vrijednost K:", best_k)
print("Najbolja prosječna točnost (CV):", "{:.3f}".format(best_score))

plot_decision_regions(X_train_n, y_train, classifier=best_knn_model)
plt.xlabel("x_1 (dob)")
plt.ylabel("x_2 (plaća)")
plt.title(f"KNN s optimalnim K={best_k}")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#optimalan K=7 a tocnost je 0.906
