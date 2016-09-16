import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import numpy as np

# Read the Breast Cancer Wisconsin dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

# Label Encoding
X = df.loc[:, 2:].values
Y = df.loc[:, 1].values
le = LabelEncoder()
Y = le.fit_transform(Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, train_size=0.8)

# cross validation
pl_le = Pipeline([('sd', StandardScaler()), ('pca', PCA(n_components=2)),
                  ('lr', LogisticRegression(penalty='l2', random_state=1))])

train_sizes, train_scores, test_scores = learning_curve(estimator=pl_le, X=X_train, y=Y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='red')
plt.show()
