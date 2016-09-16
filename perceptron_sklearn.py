from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import plot_decision_region as pt
import matplotlib.pyplot as plt

# load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
Y = iris.target
# Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Standardize the values
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)

# Perceptron model
ppn = Perceptron(eta0=0.1, n_iter=10)
ppn.fit(X_train_sd, Y_train)

# Prediction or testing
y_pre = ppn.predict(X_test_sd)
print("No of misclassifications : %d" % ((y_pre != Y_test).sum()))

# Accuracy
print("Accuracy : %.2f%%" % (accuracy_score(Y_test, y_pre)*100))

# plot the decision region
pt.plot_decision_region(X, Y, classifier=ppn)
plt.xlabel('2')
plt.ylabel('3')
plt.legend(loc = 'upper left')
plt.show()
