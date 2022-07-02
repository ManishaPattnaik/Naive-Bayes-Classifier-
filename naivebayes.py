!pip install nltk
!pip install gensim
!pip install wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics import classification_report, confusion_matrix

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  

# load the data
resume_df = pd.read_csv('resume.csv', encoding = 'latin-1')
resume_df

# data containing resume
resume_df = resume_df[['resume_text','class']]
resume_df

resume_df.head(10)

#performing exploratory data analysis
# obtain dataframe information
resume_df.info()

# check for null values
resume_df.isnull().sum()

resume_df['class'].value_counts()

resume_df['class'] = resume_df["class"].apply(lambda x:1   if x == 'flagged' else 0)
resume_df

class_1_df = resume_df[ resume_df['class']==1]
class_1_df

class_0_df = resume_df[ resume_df['class'] == 0]
class_0_df

#performing data cleaning
resume_df['resume_text'] = resume_df['resume_text'].apply( lambda  x: x.replace('\r', ''))
resume_df

# download nltk packages
nltk.download('punkt')

# download nltk packages
nltk.download("stopwords")

# Get additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use','email','com'])

# Remove stop words and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
            result.append(token)
            
    return ' '.join(result)

# Cleaned text
resume_df['cleaned'] = resume_df['resume_text'].apply(preprocess)

print(resume_df['cleaned'][0])

print(resume_df['resume_text'][0])

#data visualization
# Plot the counts of flagged vs not flagged
sns.countplot(resume_df['class'], label = 'count plot')

# plot the word cloud for text that is flagged
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000, width = 1600, height = 800, stopwords=stop_words).generate(str(resume_df[resume_df['class']==1].cleaned))

plt.imshow(wc)

plt.figure(figsize = (20,20))
wc0 = WordCloud(max_words = 1000, width = 1600, height = 800, stopwords = stop_words).generate(str(resume_df[resume_df['class']==0].cleaned))
plt.imshow(wc0)

#preparing data by applying count vectorizer
# CountVectorizer example
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)
print(vectorizer.get_feature_names())

print(X.toarray())

# Applying CountVectorier to the cleaned text
vectorizer = CountVectorizer()
countvectorizer = vectorizer.fit_transform(resume_df['cleaned'])

print(vectorizer.get_feature_names())

print(countvectorizer.toarray())

#training naive bayes classifier model

y = resume_df['class']
X.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

#assess trained model performance
# Predicting the performance on train data
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)

# classification report
print(classification_report(y_test, y_predict_test))