import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Read the Breast Cancer Wisconsin dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

# Label Encoding to convert label in column 1 to a numerical value
X = df.loc[:, 2:].values
Y = df.loc[:, 1].values
le = LabelEncoder()
Y = le.fit_transform(Y)

# Train Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

# Pipelining SS, PCA and LR
pipe_lr = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=2)), ('lr', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, Y_train)
print("Test Accuracy : %.2f%%"%(pipe_lr.score(X_test, Y_test)*100))


