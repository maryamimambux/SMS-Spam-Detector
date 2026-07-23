<div align="center">

# 📩 SMS Spam Classifier

**A machine learning web app that detects spam text messages in real time — trained on 5,000+ real SMS messages.**

[![Live Demo](https://img.shields.io/badge/📡_Live_Demo-Visit_App-0B1F17?style=for-the-badge)](https://sms-spam-classifier-iota.vercel.app/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.10-2E7D32?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-C9A227?style=flat-square)

**[🔗 Try it live → sms-spam-classifier-iota.vercel.app](https://sms-spam-classifier-iota.vercel.app/)**

</div>

---

## 📖 What is this?

Spam texts follow patterns — certain words, urgency cues, and phrasing show up again and again ("WINNER", "claim now", "free", suspicious links). A model trained on enough real examples can pick up on those patterns far more reliably than a hand-written list of blocked words.

This project cleans and vectorizes thousands of real SMS messages, trains and compares several classification algorithms, and wraps the strongest one in a live web app: paste in any message, and get an instant **Spam** / **Not Spam** verdict with a confidence score.

> Paste any SMS or text → the model cleans it (tokenize → strip punctuation/stopwords → stem), vectorizes it with TF-IDF, and predicts **Spam** or **Not Spam**, based on patterns learned from thousands of labeled messages.

---

## ✨ Features

- 🧠 **Trained ML model** (Multinomial Naive Bayes over TF-IDF features, chosen after comparing 11 algorithms)
- 🌐 **Live interactive web app** built with Flask
- 📊 **Evaluated properly** — precision prioritized over raw accuracy, since spam datasets are imbalanced
- 🎨 **Clean, custom UI** — confidence meter, example messages, no default Bootstrap look
- ☁️ **Deployed for free on Vercel**, fully reproducible via a pinned `requirements.txt`

---

## 🖥️ Try It Yourself

<div align="center">

**👉 [sms-spam-classifier-iota.vercel.app](https://sms-spam-classifier-iota.vercel.app/) 👈**

</div>

Paste in any SMS or message text, then click **Check message**.

---

## 🛠️ How It Was Built

| Stage | What Happened |
|---|---|
| **1. Data** | [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 real, labeled SMS messages |
| **2. Cleaning** | Dropped unused columns, removed duplicate messages, encoded labels (`ham` → 0, `spam` → 1) |
| **3. Text preprocessing** | Lowercased → tokenized → kept alphanumeric tokens → removed stopwords/punctuation → Porter-stemmed |
| **4. Feature engineering** | TF-IDF vectorization (`max_features=3000`) |
| **5. Model comparison** | Benchmarked 11 algorithms (Naive Bayes variants, SVM, Random Forest, AdaBoost, XGBoost, and more) on accuracy **and** precision |
| **6. Model selection** | **Multinomial Naive Bayes** — best precision of any single model, meaning almost no legitimate message gets flagged as spam |
| **7. Deployment** | Flask app, version-pinned dependencies, bundled NLTK data, deployed on Vercel |

**Final model performance (held-out test set):**

| Metric | Score |
|---|---|
| Accuracy | 95.9% |
| Precision | 100% |
| False positives (ham flagged as spam) | 0 |

> Precision was the priority metric here: in a spam filter, a false positive (a real message wrongly blocked) is far more costly than a false negative (a spam message slipping through).

---

## ⚙️ Tech Stack

`Python` · `pandas` · `NLTK` · `scikit-learn` · `Flask` · `Vercel`

---

## 🚀 Run It Locally

```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
sms-spam-classifier/
├── app.py                          # Flask app
├── model.pkl                       # Trained Multinomial Naive Bayes model
├── vectorizer.pkl                  # Fitted TF-IDF vectorizer
├── nltk_data/                      # Bundled tokenizer/stopwords data (no runtime download)
├── requirements.txt                # Pinned dependencies
├── vercel.json                     # Vercel function config
├── sms_spam_classifier_final.ipynb # Full training notebook (EDA → model comparison → export)
└── templates/
    └── index.html                  # Frontend
```

---

<div align="center">

### 👤 About

**[Your Name]**

[![GitHub](https://img.shields.io/badge/GitHub-your--username-181717?style=flat-square&logo=github)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Your_Name-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/your-profile)

*Built as a hands-on machine learning project — from raw text data to a live deployed app.*

</div>
