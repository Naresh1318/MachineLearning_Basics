from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Get data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
Y = iris.target

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=0)

# Standardize
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# model
svc = SVC(kernel='linear', C=1.0, random_state=0)
svc.fit(X_train_std, Y_train)

Y_pred = svc.predict(X_test_std)

# predict
print("Errors : %d" % (Y_test != Y_pred).sum())
