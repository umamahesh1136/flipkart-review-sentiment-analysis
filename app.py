import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------
# Page title
# ---------------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🛒")

st.title("🛒 Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to check if it is **Positive** or **Negative**.")

# ---------------------------------
# Download NLTK resources
# ---------------------------------
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

download_nltk()

# ---------------------------------
# Load model and vectorizer
# ---------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")       # Logistic Regression
    vectorizer = joblib.load("tfidf_vectorizer.pkl") # TF-IDF
    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------------
# Text preprocessing
# ---------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ---------------------------------
# User input
# ---------------------------------
review = st.text_area(
    "✍️ Enter Review",
    placeholder="Example: The shuttle quality is very poor and breaks easily..."
)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:
        cleaned_review = clean_text(review)
        review_vector = vectorizer.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]
probability = model.predict_proba(review_vector)[0][1]

if probability >= 0.6:
    st.success(f"✅ Positive Review ({probability*100:.1f}% confidence)")
elif probability <= 0.4:
    st.error(f"❌ Negative Review ({(1-probability)*100:.1f}% confidence)")
else:
    st.warning("⚠️ Sentiment unclear. Please enter a more detailed review.")

# ===============================
# Footer
# ===============================
st.divider()
st.markdown("""
<div class="footer">
    Built using TF-IDF + Linear SVM • Deployed with Streamlit  
    <br>
    Sentiment Analysis on Real-time Flipkart Reviews
</div>

""", unsafe_allow_html=True)
