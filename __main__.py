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
