# Signal Check — SMS Spam Classifier Web App

A small Flask web app that wraps the TF-IDF + Multinomial Naive Bayes
spam classifier from `sms_spam_classifier_final.ipynb` in a browser UI.

## What's included

```
sms_spam_app/
├── app.py              # Flask backend (loads vectorizer.pkl + model.pkl)
├── templates/
│   └── index.html       # Web UI (single page, no build step)
├── requirements.txt
├── vectorizer.pkl        # your trained TfidfVectorizer
├── model.pkl              # your trained MultinomialNB model
└── README.md
```

## 1. Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will download a few small NLTK data packages
(`punkt`, `punkt_tab`, `stopwords`) automatically — this needs an
internet connection once, after which they're cached locally.

## 2. Run it

Make sure `vectorizer.pkl` and `model.pkl` sit next to `app.py`
(they're already placed there), then:

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## 3. How it works

- The frontend (`templates/index.html`) posts the raw message text to
  `POST /predict` as JSON: `{"message": "..."}`.
- `app.py` runs the **exact same** `transform_text()` cleaning steps used
  during training (lowercase → tokenize → keep alphanumeric → drop
  stopwords/punctuation → Porter stemming), so the vectorizer sees text
  in the same shape it was fitted on.
- The cleaned text is vectorized with the saved `tfidf` vectorizer and
  classified with the saved `MultinomialNB` model.
- The JSON response includes the label (`Spam` / `Not Spam`), a
  confidence percentage (`predict_proba`), and the cleaned token string
  (shown in the UI for transparency).

## 4. Deploying to Vercel

This app is set up to deploy on Vercel with zero extra config — Vercel
auto-detects the `app` Flask instance in `app.py` and turns it into a
Vercel Function. NLTK's tokenizer/stopwords data is bundled in the
`nltk_data/` folder (English-only, trimmed to ~300KB) so nothing needs
to be downloaded at cold start.

**Option A — Vercel CLI**

```bash
npm i -g vercel        # one-time install
cd sms_spam_app
vercel                 # first deploy, follow the prompts
vercel --prod          # promote to production
```

**Option B — Git + Vercel dashboard**

1. Push this `sms_spam_app` folder to a GitHub (or GitLab/Bitbucket) repo.
2. Go to [vercel.com/new](https://vercel.com/new) and import that repo.
3. Vercel detects Python/Flask automatically — no build command or
   output directory needed. Click **Deploy**.
4. You'll get a live URL like `https://your-project.vercel.app`.

**Notes specific to this project**

- `vercel.json` sets `maxDuration: 30` for `app.py` — cold starts that
  load scikit-learn + NLTK can take a couple seconds, so this gives
  some headroom versus the default timeout.
- `requirements.txt` pins exact versions (`scikit-learn==1.8.0`,
  `nltk==3.10.0`, `flask==3.1.3`) to match the environment the pickles
  were saved/re-fit in. If you retrain the model with a different
  scikit-learn version, update this pin to match, or you'll get
  `InconsistentVersionWarning` (usually harmless) or, in worse cases,
  an unpickling error.
- Every request runs on a fresh/serverless instance, so nothing is
  stored between requests — this matches the app's current stateless
  design (no history, no database).
- If you later add real static assets (custom CSS/JS/images) instead
  of inline `<style>`/`<script>`, put them in a `public/` folder at
  the project root — Vercel serves that via its CDN. Flask's own
  `static/` folder is not used for that on Vercel.

## 5. Deploying elsewhere (alternative)

For a quick public demo, you can run this behind `gunicorn`:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

Then deploy to any platform that runs a Python web service (Render,
Railway, Fly.io, a VPS, etc.). Just make sure `vectorizer.pkl` and
`model.pkl` are included in the deployment and NLTK data is either
pre-downloaded in the build step or downloadable at first boot.

## Notes

- The model was trained with `scikit-learn` and pickled with `pickle`.
  If you retrain with a very different scikit-learn version, re-save
  fresh `.pkl` files to avoid version-mismatch warnings.
- This demo doesn't persist history — every check is stateless.
