import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report,accuracy_score, precision_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a)---------------------------------------------------------------------------------------------------------
plt.scatter(X_train[:,0],X_train[:,1], c="Blue",label='Train')
plt.scatter(X_test[:,0],X_test[:,1], marker="x",c="Red",label='Test')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#b)---------------------------------------------------------------------------------------------------------
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)


#c)---------------------------------------------------------------------------------------------------------
theta0 = LogRegression_model.intercept_[0]
theta1, theta2 = LogRegression_model.coef_[0]
print(f"Parametars of model: theta0 = {theta0:.3f}, theta1 = {theta1:.3f}, theta2 = {theta2:.3f}")
print("Intercept",LogRegression_model.intercept_)
print("Coef",LogRegression_model.coef_)
x_vals = np.linspace(-4,4)
y_vals = -(theta0 + theta1 * x_vals) / theta2

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.plot(x_vals, y_vals,'k--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#d)---------------------------------------------------------------------------------------------------------
y_test_p = LogRegression_model.predict(X_test)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print(classification_report(y_test , y_test_p))
print ("Accuracy : " , accuracy_score ( y_test , y_test_p ) )
print("Precision : ", precision_score (y_test, y_test_p))
print("Recall: ", recall_score(y_test, y_test_p))

#e)---------------------------------------------------------------------------------------------------------
X_false = []
for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        X_false.append([X_test[i, 0], X_test[i, 1]])

X_false = np.array(X_false)

plt.scatter(X_test[:,0], X_test[:, 1], color='green',label='Good')
plt.scatter(X_false[:,0], X_false[:,1],color="black", label='Bad')
plt.legend()
plt.show()
