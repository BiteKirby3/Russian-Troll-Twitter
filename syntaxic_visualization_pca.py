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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

tw_ = pd.read_csv('syntaxic.csv')#, nrows=10)
tw_["post_type"]=tw_["post_type"].replace(np.NaN,"TWEET")

#sns.scatterplot(x="content_length", y="punctuation_ratio1", data=tw_, hue="account_type", style="post_type")
#plt.show()

pca_tw = PCA(n_components=5)
relevant_variables = ("following","followers","updates","content_length","content_token_length","punctuation_count","punctuation_ratio1","punctuation_ratio2")


tw_pca_df = pd.DataFrame(tw_, columns=["following", "followers", "updates", "content_token_length", "punctuation_ratio1"])

pcs_tw = pca_tw.fit_transform(tw_pca_df)
sns.scatterplot(x=pcs_tw[:, 0], y=pcs_tw[:, 1], hue=tw_["account_type"], style=tw_["post_type"], data=tw_pca_df)
plt.show()

plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5"], pca_tw.explained_variance_ratio_)
plt.show()
