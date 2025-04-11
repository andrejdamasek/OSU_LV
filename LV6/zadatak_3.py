import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_decision_regions_subplot(X, y, classifier, ax, title, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)
    
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

data = pd.read_csv("Social_Network_Ads.csv")
X = data[["Age", "EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

# Postavke SVM modela
svm_settings = [
    ("rbf", 0.05, 0.01),
    ("rbf", 10, 1),
    ("rbf", 100, 50),
    ("linear", 3, "scale"),
    ("poly", 3, "scale"),
    ("sigmoid", 3, "scale"),
]

# Prikaz svih grafova
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for i, (kernel, C, gamma) in enumerate(svm_settings):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train_n, y_train)
    y_pred = model.predict(X_test_n)
    acc = accuracy_score(y_test, y_pred)
    title = f"{kernel.upper()}, C={C}, γ={gamma}\nTest acc: {acc:.2f}"
    plot_decision_regions_subplot(X_train_n, y_train, model, axs[i], title)

plt.tight_layout()
plt.show()

#za RBF veliki i mali C i γ znaci da tocnost opada  jer se dogada underfitting i overfitting,
#za isti C najbolju tocnost ima RBF,pa POLY, pa LINEAR,pa SIGMOID koji ima najlosiju tocnost