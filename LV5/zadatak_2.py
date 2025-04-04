import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,ConfusionMatrixDisplay


labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

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
        plt.scatter(x=X[y.flatten() == cl, 0],
                    y=X[y.flatten() == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor='w',
                    label=labels[cl])
    plt.legend()
    plt.title("Granica odluke i podaci za učenje")
    plt.xlabel("Bill Length (mm)")
    plt.ylabel("Flipper Length (mm)")
    plt.show()

# Učitavanje podataka
df = pd.read_csv("penguins.csv")

# Uklanjanje izostalih vrijednosti i kodiranje kategorijskih varijabli
df.drop(columns=['sex'], inplace=True)
df.dropna(inplace=True)
df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}, inplace=True)
print(df.info())

# Definiranje ulaznih i izlaznih veličina
output_variable = ['species']
input_variables = ['bill_length_mm', 'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# Podjela skupa podataka na train i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#a)---------------------------------------------------------------------------------------------------------
unique, counts_train = np.unique(y_train, return_counts=True)
unique, counts_test = np.unique(y_test, return_counts=True)
X_axis = np.arange(len(unique))
plt.bar(X_axis-0.2, counts_train,0.4,label='Train')
plt.bar(X_axis+0.2, counts_test,0.4,label='Test')
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
plt.ylabel('Number of penquins')
plt.title('Number of penguins in train and in test')
plt.legend()
plt.show()

#b)---------------------------------------------------------------------------------------------------------
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

#c)---------------------------------------------------------------------------------------------------------

print("Parametri modela:")
print(f"Intercept: {LogReg.intercept_}")
print(f"Koeficijenti: {LogReg.coef_}")

#razlika je u tome što smo u 1 zadatku imali jedan intercepr a ovjde imamo 3, i broj koeficijenata u 
# prvom zadatku je bio 2, a sada imamo 6 parametara koji se nalaze u matrici 3*2, razlog tome je sto imamo vise klasa (3)

#d)---------------------------------------------------------------------------------------------------------

plot_decision_regions(X_train, y_train, classifier=LogReg)
plt.show()
#iz grafa je moguće vidjeti da je model dobro postavio granice, te da ima malo krivih rezultata

#e)---------------------------------------------------------------------------------------------------------

y_test_pred = LogReg.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_test_pred))
disp.plot()
plt.title('Matrica zabune')
plt.show()

print(f"Točnost: {accuracy_score(y_test, y_test_pred):.2f}")
print("Izvještaj klasifikacije:")
print(classification_report(y_test, y_test_pred))

#f)---------------------------------------------------------------------------------------------------------
additional_features = ['bill_depth_mm', 'body_mass_g']
input_variables.extend(additional_features)

X_new = df[input_variables].to_numpy()
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=123)

LogReg_new = LogisticRegression()
LogReg_new.fit(X_train_new, y_train_new)

y_test_new_pred = LogReg_new.predict(X_test_new)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_test_new_pred))
disp.plot()
plt.title('Confusion matrix')
plt.show()

print("New accuracy:", accuracy_score(y_test_new, y_test_new_pred))
print("New report:",classification_report(y_test_new, y_test_new_pred))

#kada smo dodali dodatne feature tocnost se poboljsala sa  0.93 na  0.9710144927536232, te je broj pogresala pao sa 5 na 2

