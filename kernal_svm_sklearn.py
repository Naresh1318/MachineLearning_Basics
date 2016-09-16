from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plot_decision_region as pt
import numpy as np


# data
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
Y = iris.target

# test train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=0)

# std
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))

# fit
svc = SVC(C=10.0, kernel='rbf', gamma=0.01, random_state=0)
svc.fit(X_train_std, Y_train)

# pedict
Y_pred = svc.predict(X_test_std)
print("Accuracy : %.2f"%(accuracy_score(Y_pred, Y_test)*100))
pt.plot_decision_region(X_combined_std, Y_combined, classifier=svc)
plt.xlabel('p_l')
plt.ylabel('p_w')
plt.legend(loc='upper left')
plt.show()
