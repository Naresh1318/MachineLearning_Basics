from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def tokenizer(text):
    '''Used to split the text into words'''
    return text.split()

def tokenizer_porter(text):
    '''Stems the words using the Porter stemmer algorithm'''
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def generate_csv():
    import pyprind
    import os
    import pandas as pd
    import numpy as np
    import re

    np.random.seed(0)

    # Progress Bar with 50000 iterations
    pbar = pyprind.ProgBar(50000)

    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = './aclImdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as infile:
                    txt = infile.read()
                    df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()

    df.columns = ['review', 'sentiment']

    # Shuffling the data
    df = df.reindex(np.random.permutation(df.index))

    # applying preprocessing cleaning to the data

    def preprocessor(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) + \
               ' '.join(emoticons).replace('-', '')
        return text

    df['review'] = df['review'].apply(preprocessor)
    # convert to csv file
    # df.to_csv('./movie_data.csv', index=False)
    return df


df = generate_csv()
# define the stop words
stop = stopwords.words('english')

# Import the movie review database
# df = pd.read_csv('movie_data.csv', header=None)

# Train test split
X_train = df.loc[:25000, 'review'].values
Y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
Y_test = df.loc[25000:, 'sentiment'].values

# using the tfidf algo
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}
              ]

# define the pipeline
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])

# Using gridserach to find the right parameters
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy', cv=5, verbose=1,
                           n_jobs=-1)
gs_lr_tfidf.fit(X_train, Y_train)

# print best parameters
print('Best parameters set : %s '% gs_lr_tfidf)

# print best train acc
print('Best train Accuracy : %.3f'% gs_lr_tfidf.best_score_)

# print best test acc
clf = gs_lr_tfidf.best_estimator_
print ('Test Accuracy : %.3f' %clf.score(X_test, Y_test))
