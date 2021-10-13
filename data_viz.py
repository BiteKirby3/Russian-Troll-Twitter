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



tw_ = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob('data/IRAhandle_tweets_*.csv')], ignore_index = True)


pca_tw = PCA(n_components=3)

tw_pca_df = pd.DataFrame(tw_, columns=["following", "followers", "updates"])

pcs_tw = pca_tw.fit_transform(tw_pca_df)
sns.scatterplot(x=pcs_tw[:, 0], y=pcs_tw[:, 1], hue=tw_["account_type"], style=tw_["post_type"], data=tw_pca_df)
plt.show()

plt.bar(["Axe 1", "Axe 2", "Axe 3"], pca_tw.explained_variance_ratio_)
plt.show()
