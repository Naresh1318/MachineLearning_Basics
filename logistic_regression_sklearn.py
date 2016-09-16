from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# loading the data sets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
Y = iris.target

# Split train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)

# Logistic Model
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_sd, Y_train)

# Prediction
Y_pred = lr.predict(X_test_sd)
print("Errors : %d" % (Y_test != Y_pred).sum())

# Accuracy
print("Accuracy : %.2f%%" % (accuracy_score(Y_pred, Y_test) * 100))

# Example to find the classification probability
print(lr.predict_proba(X_test[40, :]))
