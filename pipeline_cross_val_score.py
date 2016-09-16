import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

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
pl_le = Pipeline([('sd', StandardScaler()), ('pca', PCA(n_components=2)), ('lr', LogisticRegression(random_state=1))])
scores = cross_val_score(estimator=pl_le, X=X_train, y=Y_train, n_jobs=1, cv=10)

print(scores)