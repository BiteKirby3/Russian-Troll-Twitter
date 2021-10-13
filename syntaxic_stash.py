import pandas as pd
import numpy as np
import glob
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
t = TweetTokenizer()

# df = pd.concat([pd.read_csv(f) for f in glob.glob('data/IRAhandle_tweets_*.csv')], ignore_index = True)
tw = pd.read_csv('data/IRAhandle_tweets_1.csv', nrows=1000)

# Removing stop words
nltk.download('punkt')
nltk.download('stopwords')
print(set(stopwords.words('Russian')))

all_stopwords = stopwords.words('english')


# tw['content_splitted'] = tw['content'].apply(lambda x: word_tokenize(x))
# tw['content_splitted_without_stopwords'] = tw['content_splitted'].apply(lambda text_token: remove_stop_word(text_token, ))
# Testing
# tw['content_splitted_without_stopwords'] = tw.apply(lambda row: row.content_splittedremove_stop_word(text_token, ))


def tokenize_remove_stop_word(row):
    text_token = t.tokenize(row['content'])
    stop_words = set()
    if row['language'] in ("English", "Russian", "French", "Italian"):
        stop_words = set(stopwords.words(row.language))

    return [word for word in text_token if not word in stop_words]


# to test
tw['content_'] = tw.apply(tokenize_remove_stop_word, axis=1)
# Appliquer à chaque row après


tw['content_']

#list_of_all_words = list(itertools.chain.from_iterable(list(tw['content_'])))
test_phrase = "As much as I hate promoting CNN article, here they are admitting EVERYTHING Trump said about PR relief two days ago. https://t.co/tZmSeA48oh"

list_of_all_words = word_tokenize(test_phrase)
word_occurences = pd.Index(list_of_all_words).value_counts().sort_values(ascending=False)




## STASH vvvv


# Right Troll, id 8
nltk.download('punkt')
test_phrase = "As much as I hate promoting CNN article, here they are admitting EVERYTHING Trump said about PR relief two days ago. https://t.co/tZmSeA48oh"
text_tokens = word_tokenize(test_phrase)

# test
tokenize_remove_stop_word(test_phrase, "English")

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

all_stopwords = stopwords.words('english')

text_tokens = word_tokenize(test_phrase)
tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
print(tokens_without_sw)
