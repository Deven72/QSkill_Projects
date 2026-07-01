# Signal — Text Sentiment Analyzer (Django + TextBlob)

A polished Django web app that classifies user-entered text as **Positive**,
**Negative**, or **Neutral**, and displays **polarity** and **subjectivity**
scores using TextBlob.

## Features
- Classifies text sentiment as Positive / Negative / Neutral
- Displays polarity score (-1.0 to +1.0) on a visual gauge
- Displays subjectivity score (0.0 to 1.0) — factual vs. opinionated
- Sentence-by-sentence sentiment breakdown for multi-sentence input
- Word/sentence count metadata
- Clean dark "signal analyzer" themed UI, fully responsive

## Setup Instructions (Windows / VS Code)

### 1. Create a project folder and place the files

### 2. Create & activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download TextBlob's language corpora (one-time setup)
```bash
python -m textblob.download_corpora
```

### 5. Run database migrations (Django requires this even without custom models)
```bash
python manage.py migrate
```

### 6. Start the development server
```bash
python manage.py runserver
```

### 7. Open in your browser
Go to: **http://127.0.0.1:8000/**

Type any sentence or paragraph and click **RUN ANALYSIS** to see:
- Overall sentiment verdict (Positive / Negative / Neutral)
- Polarity score with visual gauge
- Subjectivity score with visual gauge
- Sentence-level breakdown (if multiple sentences entered)

## Project Structure
```
sentiment_project/
├── manage.py
├── requirements.txt
├── sentimentapp/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── analyzer/              # Main app (sentiment logic)
│   ├── views.py           # TextBlob analysis logic
│   ├── urls.py
│   └── models.py
└── templates/
    └── analyzer/
        └── index.html     # Styled frontend
```

## How It Works
1. User submits text via a POST form
2. `TextBlob(text).sentiment` returns:
   - **polarity**: -1.0 (very negative) to +1.0 (very positive)
   - **subjectivity**: 0.0 (factual) to 1.0 (opinionated)
3. Polarity is thresholded into Positive (>0.1) / Negative (<-0.1) / Neutral (between)
4. Results render dynamically with gauges and per-sentence breakdown

## Notes
- No database models are required — this app is stateless (analysis only, no storage)
- To deploy, set `DEBUG = False` and configure `ALLOWED_HOSTS` in `settings.py`
