import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re, csv, nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("amazon_alexa_data.csv")
#Removing handle null values
data.dropna(inplace=True)
#Tokenizing words
tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+")
for x in data['verified_reviews']:
     x = tokenizer.tokenize(x)
#Converting words to lower case
data['verified_reviews'] = data['verified_reviews'].str.lower()
#Removing punctuations
data['verified_reviews'] = data['verified_reviews'].str.translate(str.maketrans('', '', string.punctuation))
#Removing stop words
stop_words = set(stopwords.words('english'))
data['verified_reviews'] = data['verified_reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
#Stemming or Lemmatizing the words
lmtzr = WordNetLemmatizer()
for word in data['verified_reviews']:
    word = lmtzr.lemmatize(word)

#Tfid Vectorizer
# Transform features
vectorizer = TfidfVectorizer()
X = data.verified_reviews
X_tfidf = vectorizer.fit_transform(X)

# create target
y = data.rating

# split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.33, random_state=42
)

#Multinomial Naive Bayes Classification
# Training classifier model 
clf = SGDClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#Computing Confusion matrix
print("Multinomial Naive Bayes Confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred))
print()
#Classification Report
print("Multinomial Naive Bayes Classification Report")
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))
accuracy_score = metrics.accuracy_score(y_test,y_pred)
print()
print("Multinomial Naive Bayes accuracy_score: ",accuracy_score)

#Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print()
#Computing Confusion matrix
print("Logistic Regression Confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred))
print()
#Classification Report
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))
accuracy_score = metrics.accuracy_score(y_test,y_pred)
print()
print("Logistic Regression accuracy_score: ",accuracy_score)


#KNN Classfication
#Finding k-value
error1= []
error2= []
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred1= knn.predict(X_train)
    error1.append(np.mean(y_train!= y_pred1))
    y_pred2= knn.predict(X_test)
    error2.append(np.mean(y_test!= y_pred2))
# plt.figure(figsize(10,5))
plt.plot(range(1,15),error1,label="train")
plt.plot(range(1,15),error2,label="test")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)

#Computing Confusion matrix
print()
print("KNN classifier Confusion Matrix")
print(metrics.confusion_matrix(y_test, y_pred))
print()
#Classification Report
print("KNN Classifictaion report")
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))
accuracy_score = metrics.accuracy_score(y_test,y_pred)
print()
print("KNN CLassifier accuracy_score: ",accuracy_score)