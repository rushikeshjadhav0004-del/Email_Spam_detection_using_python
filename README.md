# 📧 Email Spam Detection using Machine Learning

## 📌 Project Overview
This project focuses on building a **machine learning–based Email Spam Detection system** that classifies emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

The solution includes **end-to-end workflow**: data cleaning, exploratory data analysis (EDA), text preprocessing, feature engineering, model building, evaluation, and real-time prediction.

---

## 🎯 Problem Statement
Spam emails cause security risks and reduce productivity.  
The goal of this project is to **automatically detect spam emails** based on their content using supervised machine learning.

---

## 📂 Dataset
- Dataset: SMS Spam Collection Dataset
- Features:
  - `message`: Email/SMS text
  - `label`: spam / ham
  - `label_encoded`: Numerical label (0 = ham, 1 = spam)
  - `message_length`: Length of the message

---

## 🔍 Exploratory Data Analysis (EDA)
- Analyzed **spam vs ham distribution**
- Compared **message length** across classes
- Visualized data using:
  - Count plots
  - Histograms
  - Box plots
  - Word clouds
  - Top frequent words

**Key Insight:**  
Spam messages generally have distinct word patterns and different message lengths compared to ham messages.

---

## 🧹 Data Preprocessing
- Converted text to lowercase
- Removed punctuation, numbers, and stopwords
- Applied stemming
- Encoded target labels using `LabelEncoder`

---

## 🧠 Feature Engineering
- Used **TF-IDF Vectorizer** with:
  - Unigrams and bigrams (`ngram_range=(1,2)`)
  - Maximum 3000 features
- Added `message_length` as an additional numeric feature
- Combined text and numeric features for better performance

---

## 🤖 Machine Learning Model
- **Algorithm Used:** Multinomial Naive Bayes
- **Why MultinomialNB?**
  - Designed for text classification
  - Works well with frequency-based features
  - Fast and efficient

```python
MultinomialNB(alpha=0.1)
