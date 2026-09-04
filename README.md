🛡️ **Bilingual Email & SMS Spam Classifier (Hindi & English)**

An end-to-end Machine Learning web application designed to detect and filter spam across both Email and SMS communication channels in English and Hindi (Devanagari script).

📌 **Project Overview**

Spam detection often fails in multilingual environments when systems are tuned only for a single language or a single message format. This project provides a unified bilingual filter capable of classifying both short-form SMS alerts and structured email messages across Hindi and English. By combining cross-domain datasets, applying script-aware text preprocessing, training a probability-calibrated Logistic Regression model, and serving it through an interactive Streamlit UI, the system delivers real-time inference along with rigorous statistical evaluations.

**Team Member Contributions**

**Aanya Godiyal** : (Model Training, Architecture & UI)

• Designed the core machine learning pipeline using Logistic Regression.

• Built the feature extraction workflow using TfidfVectorizer (unigram + bigram text representation).

• Calibrated prediction probabilities for customizable decision thresholds.

• Managed model persistence and serialization (joblib/pickle).

• Developed the interactive two-view Streamlit dashboard (Predictor interface & Analytics view).

**Mahi Jain** : (Data Preprocessing & Statistical Analysis)

• Curated, aligned, and unified multilingual Email and SMS datasets.

• Developed the cleaning pipeline (URL/email handling, punctuation removal, regex cleaning, and UTF-8 Unicode encoding management).

• Implemented language-aware stopword strategies preserving Devanagari script integrity.

• Computed core classification statistics: Accuracy, Precision, Recall, and F1-Score.

• Constructed the Confusion Matrix to isolate True Positives, False Positives, and False Negatives.

• Performed error analysis on misclassified emails and SMS messages to diagnose model boundary errors.
