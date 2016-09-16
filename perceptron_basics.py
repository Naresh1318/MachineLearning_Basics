from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
Y = iris.target

print(X.shape[1])

# plot the values of X

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o',label = 'setosa')
plt.scatter(X[50:, 0],X[50:, 1], color = 'blue', marker='x',label = 'versicolor')
plt.xlabel('patel length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()