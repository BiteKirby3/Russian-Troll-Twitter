import pandas as pd
import numpy as np
import glob
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
import string
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')


#tw = pd.read_csv('data/IRAhandle_tweets_1.csv')#, nrows=10)
tw = pd.concat([pd.read_csv(f) for f in glob.glob('data/IRAhandle_tweets_*.csv')], ignore_index = True)

tw["post_type"]=tw["post_type"].replace(np.NaN,"TWEET")

def tokenize_remove_stop_word(row):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    text_token = TweetTokenizer().tokenize(row['content'])
    stop_words = set()
    if row['language'] in ("English", "Russian", "French", "Italian"):
        stop_words = set(stopwords.words(row.language))
    return [word for word in text_token if not word in stop_words]

def extract_punctuation(x):
    tknzr = TweetTokenizer(strip_handles=False, reduce_len=False)
    text_token = TweetTokenizer().tokenize(x['content'])
    return [word for word in text_token if word in string.punctuation]

def extract_punctuation_ratio(x):
    tknzr = TweetTokenizer(strip_handles=False, reduce_len=False)
    text_token = TweetTokenizer().tokenize(x['content'])
    return len(list([word for word in text_token if word in string.punctuation]))/len(x['content'])

def extract_punctuation_ratio1(x):
    tknzr = TweetTokenizer(strip_handles=False, reduce_len=False)
    text_token = TweetTokenizer().tokenize(x['content'])
    return len(list([word for word in text_token if word in string.punctuation]))/len(x['content'])
def extract_punctuation_ratio2(x):
    tknzr = TweetTokenizer(strip_handles=False, reduce_len=False)
    text_token = TweetTokenizer().tokenize(x['content'])
    return len(list([word for word in text_token if word in string.punctuation]))/len(x['content_token'])

tw['content_length'] = tw.apply(lambda x: len(list(x['content'])), axis=1)
tw['content_token'] = tw.apply(tokenize_remove_stop_word, axis=1)
tw['content_token_length'] = tw.apply(lambda x: len(list(x['content_token'])), axis=1)
tw['punctuation'] = tw.apply(extract_punctuation, axis=1)
tw['punctuation_count'] = tw.apply(lambda x: len(list(x['punctuation'])), axis=1)
tw['punctuation_ratio1'] = tw.apply(extract_punctuation_ratio1, axis=1)
tw['punctuation_ratio2'] = tw.apply(extract_punctuation_ratio2, axis=1)
#w['punctuation_count'] = tw.apply(lambda x : len(extract_punctuation(tw['content'])), axis=1)
#"content_length","content_token","content_token_length","punctuation","punctuation_count","punctuation_ratio1","punctuation_ratio2"

tw.to_csv('syntaxic_full.csv')
print("syntaxic extraction done")