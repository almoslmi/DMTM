
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import string
import math
import nltk
#nltk.download('popular')
import scipy.sparse as sp
from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPParser
from nltk.stem import PorterStemmer # for stemming if need be
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


train_df = pd.read_csv('project_2_train/data_1_train.csv')
train_df.columns = ['example_id', 'text', 'aspect_term', 'term_location', 'class']

# proportion of each sentiment
classes, counts = np.unique(train_df['class'], return_counts = True)
prop = counts/ sum(counts) # positive > negative > neutral

# replacing [comma] in text with ','
train_df['text'] = train_df['text'].apply(lambda str : str.replace("[comma]", ","))

# tokenization
train_df['tokens'] = train_df['text'].apply(nltk.word_tokenize)

# removing stopwords and puctuation
stop = set(stopwords.words('english'))
train_df['tok_wo_stop'] = train_df['tokens'].apply(lambda val : [x for x in val if (x not in stop and x not in string.punctuation)])

# tagging
#train_df['tagged'] = train_df['tok_wo_stop'].apply(nltk.pos_tag) #taking too much time, think of better way if possible

# Bag of Words
train_df['token_list'] = train_df['tok_wo_stop'].apply(lambda x : " ".join(x))


# In[40]:


# converting text to features
vectorizer = CountVectorizer(analyzer = "word",
                            tokenizer = None,
                            preprocessor = None,
                            stop_words = None,
                            min_df = 2)
sample = train_df[['token_list', 'aspect_term']]
features = sp.hstack(sample.apply(lambda col: vectorizer.fit_transform(col))) #no polarity of aspect taken into account yet!
features = features.toarray()

# splitting features into training and test feature set (90:10)
train_end = math.floor(0.9*(train_df['token_list'].size))
train_features = features[:train_end]
test_features = features[train_end:]


# In[41]:


forest = RandomForestClassifier(n_estimators = 100)
#forest = GaussianNB()
forest = forest.fit(train_features, train_df.loc[ : train_end - 1 , 'class'])


# In[42]:


result = forest.predict(test_features)
print(((result == train_df.loc[train_end :  , 'class']).sum()) / result.size)

