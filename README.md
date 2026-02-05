# Flipkart Review Sentiment Analyzer 🛒

This project is a simple **Sentiment Analysis web application** that predicts whether a Flipkart product review is **Positive** or **Negative**.

The app is built using **Machine Learning and Streamlit** and is deployed on **Hugging Face Spaces**.

---

## 📌 Project Overview

* Uses customer reviews scraped from Flipkart
* Converts review text into numerical features using **TF-IDF**
* Trains a **Logistic Regression** model for sentiment classification
* Displays prediction along with **confidence score**
* Provides real-time sentiment analysis through a Streamlit web app

---

## 🧠 Machine Learning Approach

* **Text Preprocessing**

  * Lowercasing
  * Removing special characters
  * Stopword removal
  * Lemmatization

* **Feature Extraction**

  * TF-IDF Vectorization

* **Model Used**

  * Logistic Regression

* **Evaluation Metric**

  * F1-Score

---

## 🖥️ Web Application

* Built using **Streamlit**
* User enters a review text
* App predicts:

  * Positive or Negative sentiment
  * Confidence percentage

---

## 📂 Project Structure

```
flipkart-sentiment-analyzer/
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Deployment

The application is deployed using **Hugging Face Spaces** with Streamlit as the SDK.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* NLTK
* Pandas
* NumPy

---

## 📊 Example Input

```
The shuttle quality is very poor and breaks easily.
```

**Output:**
❌ Negative Review (Confidence shown)

---

## 📌 Future Improvements

* Use advanced models like BERT
* Add sentiment explanation using keywords
* Improve handling of very short reviews

---

## 👤 Author

**Uma Mahesh**
Aspiring Data Analyst & Cricket Analyst


