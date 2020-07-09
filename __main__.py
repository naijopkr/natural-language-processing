"""
Glossary:
bow: Bag of words
tfidf: Term Frequency - Inverse Document Frequency
"""

import nltk
import pandas as pd

# stopwords needs to be download using nltk.download()
from nltk.corpus import stopwords

messages = pd.read_csv(
    './data/sms.tsv',
    sep='\t',
    names=['label', 'message']
)
messages.head()
messages.info()
messages.describe()

messages.groupby('label').describe()

# Feature engineering
messages['length'] = messages['message'].apply(len)
messages.head()

# Data visualization
import seaborn as sns

sns.distplot(messages['length'], bins=50)
messages['length'].describe()

# Max length is 910, let's see what's in there
messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column='length', by='label', bins=50, figsize=(12,4))

# Text pre-processing
import string

def text_process(msg):
    """
    Takes in a string of characters,
    then preforms the following:
    1. Removes punctuation
    2. Removes stopwords
    3. Returns list of words
    """
    no_punctuation = ''.join(
        [char for char in msg if char not in string.punctuation]
    )

    word_list = [
        word for word in no_punctuation.split()
            if word.lower() not in stopwords.words('english')
    ]

    return word_list

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
len(bow_transformer.vocabulary_) # 11425

messages_bow = bow_transformer.transform(messages['message'])
messages_bow.shape # (5572, 11425)
messages_bow.nnz # Non zero: 50548

sparsity = (
    100.0 * messages_bow.nnz /
        (messages_bow.shape[0] * messages_bow.shape[1])
) # 0.0794

#TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit(messages_bow)
tfidf.idf_[bow_transformer.vocabulary_['u']] # 3.28
tfidf.idf_[bow_transformer.vocabulary_['university']] # 8.52

messages_tfidf = tfidf.transform(messages_bow)
messages_tfidf.shape # (55572, 11425)

# Training model
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

# Prediction
y_true = messages['label']
y_pred = spam_detect_model.predict(messages_tfidf)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

cr = classification_report(y_true, y_pred)
print(cr)

[tn, fp], [fn, tp] = confusion_matrix(y_true, y_pred)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (fp + fn + tp + tn)


# Use train test split
from sklearn.model_selection import train_test_split

X = messages['message']
y = messages['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train) # 4457
len(X_test) # 1115
len(y_train) # 4457
len(y_test) # 1115

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    # strings to token counts
    ('bow', CountVectorizer(analyzer=text_process)),
    # integer counts to weighted TF-IDF scores
    ('tfidf', TfidfTransformer()),
    # train on TF-IDF vectors w/ Naive Bayes Classifier
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

cr_pipe = classification_report(y_test, y_pred)
print(cr_pipe)

[tn, fp], [fn, tp] = confusion_matrix(y_test, y_pred)
accuracy = (tp + tn) / (tn + fp + fn + tp) # 0.96
