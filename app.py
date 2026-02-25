import streamlit as st
import pickle
import numpy as np
import re

# ---------------------------------
# PAGE SETTINGS
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

#bcg_color = "#f0f2f6"
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#5f6dfb,#d66efd);
}

/* White container */
.block-container {
    background-color: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#5f6dfb,#d66efd);
}

            
/* White container */
.block-container {
    background-color: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
    margin-top:80px;
    margin-bottom:60px;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------
# LOAD MODELS
# ---------------------------------
@st.cache_resource
def load_models():
    with open("w2v_model.pkl", "rb") as f:
        w2v_model = pickle.load(f)

    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)

    return w2v_model, svm_model

w2v_model, svm_model = load_models()

# ---------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text.split()

# ---------------------------------
# CONVERT TEXT TO VECTOR
# ---------------------------------
def get_average_vector(tokens):
    vectors = []

    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)

# ---------------------------------
# UI DESIGN
# ---------------------------------
st.title("📰 Fake News Detection System")
st.markdown("### Enter a news article to check if it is Fake or Real")

st.markdown("---")


# ----------- NEW TITLE CONTAINER -----------
news_title = st.text_input(
    "📝 Enter News Title:"
)

# ----------- CONTENT CONTAINER -----------
news_input = st.text_area(
    "✍ Paste your news content below:",
    height=200
)

# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------
if st.button("🔍 Detect News"):

    if news_input.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:
        # Combine title + content
        full_text = news_title + " " + news_input
        
        tokens = clean_text(full_text)
        vector = get_average_vector(tokens).reshape(1, -1)

        prediction = svm_model.predict(vector)[0]

        st.markdown("---")

        if prediction == 1:
            st.success("✅ This News is REAL")
        else:
            st.error("🚨 This News is FAKE")
            