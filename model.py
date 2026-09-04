import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


df = pd.read_csv(r"C:\Users\aanya\OneDrive\Desktop\spamfilter\SpamDataset\combined_dataset.csv", 
                 encoding='utf-8-sig')
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(df['Text']) 
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

def predictSpam(text):
    vecText = tfidf.transform([text])
    prediction=model.predict(vecText)
    probab=model.predict_proba(vecText)
    if prediction==1:
        confidence=probab[0][1]*100
        return {"label": "SPAM", "confidence": round(confidence, 2)}
    else:
        confidence=probab[0][0]*100
        return {"label": "HAM", "confidence": round(confidence, 2)}
    
def is_suspicious_sender(sender):
    suspicious_patterns = [r'\d{5,}', r'[^a-zA-Z0-9@.]']
    for pattern in suspicious_patterns:
        if re.search(pattern, sender):
            return True
    return False

def classifyEmail(sender,subject,body):
    subject_vec = tfidf.transform([subject])
    subject_prob = model.predict_proba(subject_vec)
    body_vec = tfidf.transform([body])
    body_prob = model.predict_proba(body_vec)
    spam_score = (subject_prob[0][1] * 0.3) + (body_prob[0][1] * 0.7)

    if is_suspicious_sender(sender):
        spam_score += 0.30
    is_spam = spam_score > 0.40
    if is_spam ==1:
        return {"label": "SPAM","confidence":round(spam_score*100,2)}
    else:
        return {"label": "HAM","confidence":round(spam_score*100,2)}


y_pred = model.predict(X_test)

def get_stats():
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred) 
    }

def get_confusion_matrix_data():

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return [tn, fp, fn, tp]