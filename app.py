"""
SMS Spam Classifier - Flask Web App
------------------------------------
Loads the pre-trained TF-IDF vectorizer (vectorizer.pkl) and
MultinomialNB model (model.pkl) produced in sms_spam_classifier_final.ipynb,
and serves a simple web UI + JSON API for classifying SMS/text messages
as Spam or Ham (not spam).
"""

import os
import pickle
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request, jsonify

# ------------------------------------------------------------------
# NLTK data setup.
# The required data (punkt_tab tokenizer + english stopwords) ships
# inside the ./nltk_data folder in this repo, so no runtime download
# is needed - this matters on serverless platforms like Vercel, where
# outbound downloads on cold start are slow/unreliable and the
# filesystem is read-only outside /tmp.
# ------------------------------------------------------------------
LOCAL_NLTK_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
if LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, LOCAL_NLTK_DATA)

for pkg, sub in [("punkt_tab", "tokenizers"), ("stopwords", "corpora")]:
    try:
        nltk.data.find(f"{sub}/{pkg}")
    except LookupError:
        # Fallback for local dev if the bundled data is missing for some reason.
        nltk.download(pkg)

ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)

# ------------------------------------------------------------------
# Same transform_text() used during training - MUST match exactly,
# otherwise the vectorizer's vocabulary won't line up with new input.
# ------------------------------------------------------------------
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # remove stopwords & punctuation
    tokens = [t for t in tokens if t not in STOPWORDS and t not in PUNCT]

    # stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


# ------------------------------------------------------------------
# Load the trained artifacts (absolute paths so this works no matter
# what the process's working directory is, e.g. on Vercel)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a message to classify."}), 400

    transformed = transform_text(message)
    vector = tfidf.transform([transformed])

    prediction = int(model.predict(vector)[0])          # 0 = ham, 1 = spam
    proba = model.predict_proba(vector)[0]
    confidence = float(proba[prediction]) * 100

    result = {
        "label": "Spam" if prediction == 1 else "Not Spam",
        "is_spam": bool(prediction == 1),
        "confidence": round(confidence, 2),
        "cleaned_text": transformed,
    }
    return jsonify(result)


if __name__ == "__main__":
    # host=0.0.0.0 so it's reachable if you deploy/containerize this later
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
