import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
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
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

data = pd.read_csv("Social_Network_Ads.csv")
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=10)
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

param_grid = {
    'C': [0.5,1,2,3,4,5,10,20,50, 100, 1000],
    'gamma': [10, 1, 0.7, 0.5, 0.3, 0.1, 0.01, 0.01]
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)

best_model = grid_search.best_estimator_
y_test_p = best_model.predict(X_test_n)


print("Najbolji parametri:", grid_search.best_params_)
print("Test točnost:", accuracy_score(y_test, y_test_p))

plot_decision_regions(X_train_n, y_train, classifier=best_model)
plt.title(f"RBF SVM (C={grid_search.best_params_['C']}, γ={grid_search.best_params_['gamma']})")

#Najbolji parametri: {'C': 1, 'gamma': 1}
#Test točnost: 0.925