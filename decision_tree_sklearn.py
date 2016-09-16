import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import plot_decision_region as pt
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

# data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
Y = iris.target

# split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=0)



# model
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dt.fit(X_train, Y_train)

# predict
Y_pred = dt.predict(X_test)
print("Accuracy : %.2f"%(accuracy_score(Y_pred, Y_test)*100))

# plot
X_comd = np.vstack((X_train, X_test))
Y_comd = np.hstack((Y_train,Y_test))
pt.plot_decision_region(X_comd, Y_comd, classifier=dt)
plt.xlabel("P L")
plt.ylabel("P W")
plt.show()

# decision tree graph
export_graphviz(dt, out_file='dt.dot', feature_names=['PL', 'PW'])