import nltk
import nltk.corpus
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
import seaborn as sns

my_dir = r"C:\Users\YUVRAJ\Documents\Data Science\text_topics"
filelist = []
filesList = []
os.chdir( my_dir )

loc = r"C:\Users\YUVRAJ\Documents\Data Science\text_topics"
os.chdir(loc)
filelist = os.listdir()

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f,'r') as myfile:
        data.append(myfile.read())
        
df = pd.DataFrame(data)
df.to_csv("cape_text.csv")

print (df.shape)

labels = pd.read_csv(r"C:\Users\YUVRAJ\Documents\Data Science\target.csv")
n1 = range(1,2965)
df["filename"] = n1
final_dat = pd.merge(df,labels, on="filename")

final_dat.columns= ["messages", "S.no.", "Topic"]
import re
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
corpus =[]

for i in range(0, len(final_dat)):
    review = re.sub('[^a-zA-Z]', ' ', final_dat["messages"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = ' '.join(review)
    corpus.append(review)

cv= CountVectorizer(max_features=20000)
X = cv.fit_transform(corpus).toarray()

le = LabelEncoder()
y= le.fit_transform(final_dat["Topic"].values)

Tf_log = TfidfVectorizer(stop_words = "english").fit(corpus)
tf_features = Tf_log.transform(corpus)

# Unsupervised clustering of text documents
Tf_log_cl = TfidfVectorizer(stop_words = "english", ngram_range=(1,3)).fit(corpus)
tf_features_cl = Tf_log_cl.transform(corpus)
km = KMeans(n_clusters=3).fit(tf_features)
txt_clust = final_dat.copy()
txt_clust["Labels"]= km.labels_
sns.pairplot(txt_clust, hue="Labels")

X_tr, X_te, Y_tr, Y_te = train_test_split(tf_features, y, test_size=0.20, random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

log_mod = LogisticRegression().fit(X_train, Y_train)
log_pred = log_mod.predict(X_test)
pd.crosstab(Y_test, log_pred)
# (196+195+195)/7 83.7% accuracy
 
# Bigrame model
Tf_log_2 = TfidfVectorizer(stop_words = "english", ngram_range=[1,2]).fit(corpus)
tf_features_2 = Tf_log.transform(corpus)

X_tr2, X_te2, Y_tr2, Y_te2 = train_test_split(tf_features_2, y, test_size=0.20, random_state=0)

# unigram logistic model
tf_log_mod = LogisticRegression().fit(X_tr, Y_tr)
tf_log_pred = tf_log_mod.predict(X_te)
pd.crosstab(Y_te, tf_log_pred)
# 589/593 99.3% accuracy

# Bigram logistic model
tf_log_mod_2 = LogisticRegression().fit(X_tr2, Y_tr2)
tf_log_pred_2 = tf_log_mod.predict(X_te2)
pd.crosstab(Y_te2, tf_log_pred_2)
# 589/593 99.3% accuracy

# unigram decission tree model
tf_dec_mod = DecisionTreeClassifier().fit(X_tr, Y_tr)
tf_dec_pred = tf_dec_mod.predict(X_te)
pd.crosstab(Y_te, tf_dec_pred)
# (182+169+179)/593 89.3% accuracy

# Bigram logistic model
tf_dec_mod_2 = DecisionTreeClassifier().fit(X_tr2, Y_tr2)
tf_dec_pred_2 = tf_dec_mod.predict(X_te2)
pd.crosstab(Y_te2, tf_dec_pred_2)
# (188+169+179)/593 90.3% accuracy


# Classification using LSTM neural network

import tensorflow as tf
tf.__version__






































































