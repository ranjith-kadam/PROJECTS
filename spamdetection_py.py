# -*- coding: utf-8 -*-
"""spamDetection.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gybDoldmJ30lmbAxoeJ4EL9BPA5Rnsbf
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

uploaded_files = os.listdir()
print(uploaded_files)


filename = 'spam.csv'
data = pd.read_csv(filename, encoding='latin-1')

# Display the first few rows of the dataset
data.head()

# Rename columns if needed
data = data[['v1', 'v2']]
data.columns = ['label', 'message']


data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')


X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))


# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))