
# Flipkart Review Sentiment Analyzer 🛒

A beginner-friendly **Sentiment Analysis web application** that predicts whether a Flipkart product review is **Positive** or **Negative** using Machine Learning.

The app is built with **Logistic Regression + TF-IDF**, provides confidence scores, and is deployed on **AWS EC2 using Streamlit**.

---

## 🌐 Live Demo

🔗 **Application URL:**

```
[http://<YOUR_EC2_PUBLIC_IP>:8501](http://13.201.47.199:8501/)
```
Anyone can access this link in a browser.

---

## 📌 Project Overview

Customer reviews contain valuable insights about product quality and user experience.
This project classifies Flipkart product reviews into **positive or negative sentiment** and presents the results through a simple web interface.

---

## 🧠 Machine Learning Approach

### Text Preprocessing

* Lowercasing text
* Removing special characters
* Stopword removal
* Lemmatization

### Feature Extraction

* TF-IDF (Term Frequency–Inverse Document Frequency)

### Model Used

* Logistic Regression

### Evaluation Metric

* F1-Score

---

## 🖥️ Web Application

* Built using **Streamlit**
* User enters a product review
* The app:

  * Cleans the text
  * Converts it into TF-IDF features
  * Predicts sentiment (Positive / Negative)
  * Displays a confidence percentage

---

## 📂 Project Structure

```
flipkart-review-sentiment-analysis/
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Deployment

The application is deployed on an **AWS EC2 instance** and runs in the background using `nohup`.

### Key deployment details:

* Streamlit app runs on port **8501**
* Virtual environment used for dependency management
* Public access enabled via EC2 Security Group

### Command used to run the app:

```bash
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

---

## 📊 Example

**Input Review:**

```
It breaks easily and feels very cheap.
```

**Output:**

```
❌ Negative Review (Confidence shown)
```

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* NLTK
* Pandas
* NumPy
* AWS EC2

---

## 📌 Future Improvements

* Add neutral sentiment classification
* Use transformer models like BERT
* Improve handling of very short reviews
* Add keyword-based explanation for predictions

---

## 👤 Author

**Uma Mahesh**
Aspiring Data Analyst & Cricket Analyst
