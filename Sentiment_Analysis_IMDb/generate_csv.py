import pyprind
import os
import pandas as pd
import numpy as np
import re

np.random.seed(0)

# Progress Bar with 50000 iterations
pbar = pyprind.ProgBar(50000)

labels = {'pos' : 1, 'neg' : 0}
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile :
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
df.to_csv('./movie_data.csv', index=False)