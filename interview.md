# 🎯 YouTube Comment Sentiment Analysis — Interview Preparation Guide

> **Project:** YouTube Sentiment Insights  
> **Stack:** Python · LightGBM · TF-IDF · FastAPI · DVC · MLflow · Docker · Chrome Extension  
> **GitHub Repo:** `Kartik87580/Youtube-Sentiment-Analysis-`

---

## 📌 Table of Contents

1. [What is the Project?](#1-what-is-the-project)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How the Project Solves the Problem](#3-how-the-project-solves-the-problem)
4. [Project Architecture](#4-project-architecture)
5. [Technical Explanation — Key Functions & Code](#5-technical-explanation--key-functions--code)
6. [Interview Questions & Answers](#6-interview-questions--answers)

---

## 1. What is the Project?

**YouTube Comment Sentiment Analysis** is an end-to-end ML system that automatically classifies YouTube video comments into three sentiment classes:

| Label | Meaning | Value |
|-------|---------|-------|
| Positive | Praise, admiration, support | `1` |
| Neutral  | Informational, neither positive nor negative | `0` |
| Negative | Criticism, hate, frustration | `-1` |

The project delivers this capability via a **Chrome browser extension** that runs directly on any YouTube video page. The user clicks the extension, and it:
- Fetches comments from the YouTube Data API v3
- Sends them to a local FastAPI backend
- Displays a pie chart, word cloud, trend graph, and per-comment sentiment labels

The full ML lifecycle (data → preprocessing → training → evaluation → registration → deployment) is automated using **DVC pipelines** and **MLflow** for experiment tracking, and the backend is Dockerized and deployed via a **GitHub Actions CI/CD pipeline**.

---

## 2. The Problem It Solves

### Real-World Problem
YouTube creators, marketers, and brand managers receive **thousands of comments** per video. Manually reading and understanding audience sentiment is:
- **Humanly impossible** at scale (top videos get 10,000–1,000,000+ comments)
- **Time-consuming** and error-prone
- **Not real-time** — by the time insights are gathered, opportunity is lost

### Specific Pain Points
- A creator cannot tell if their new video is being received positively or negatively in real time
- Brand managers can't quickly gauge public opinion on their YouTube ads
- There is no quick way to spot a spike in negative comments (e.g., a PR crisis)
- No accessible tool exists as a browser extension — existing tools require CSV exports and separate dashboards

---

## 3. How the Project Solves the Problem

The solution is a **one-click in-browser analytics panel** powered by a production-grade ML model running in the background.

### Solution Flow

```
User clicks Extension ──► YouTube Data API v3
                               │
                               ▼
                     Fetch up to 200 comments
                     (with timestamps + author IDs)
                               │
                               ▼
                   FastAPI Backend (localhost:8000)
                    /predict_with_timestamps
                               │
                               ▼
              preprocess_comment() → TF-IDF → LightGBM
                               │
                               ▼
               Sentiment Labels: 1 / 0 / -1 per comment
                               │
              ┌────────────────┼─────────────────────┐
              ▼                ▼                     ▼
       Pie Chart         Word Cloud           Trend Graph
    /generate_chart   /generate_wordcloud  /generate_trend_graph
              │                                       │
              └──────────── Chrome Popup UI ──────────┘
```

### What the user sees:
- 📊 **Pie Chart** — Distribution of Positive / Neutral / Negative
- ☁️ **Word Cloud** — Most frequent meaningful words in comments
- 📈 **Trend Graph** — Monthly sentiment % change over time
- 🏷️ **Top 25 Comments** — Each comment labelled with its sentiment
- 📋 **Metrics Panel** — Total comments, unique users, avg word length, sentiment score (0–10)

---

## 4. Project Architecture

### Directory Structure

```
youtube_comment_analysis/
│
├── src/                          # ML pipeline source code
│   ├── data/
│   │   ├── data_ingestion.py     # Download + split dataset
│   │   └── data_preprocessing.py # Clean & normalize text
│   └── model/
│       ├── model_building.py     # TF-IDF + LightGBM training
│       ├── model_evaluation.py   # MLflow evaluation & logging
│       └── register_model.py     # Register model to MLflow Registry
│
├── app/
│   └── main.py                   # FastAPI backend (5 endpoints)
│
├── crome_extension/              # Chrome Extension (MV3)
│   ├── manifest.json             # Extension metadata & permissions
│   ├── popup.html                # Extension UI
│   ├── popup.js                  # Extension logic
│   └── content.js                # Content script
│
├── notebooks/                    # Experimentation notebooks (1–8)
├── data/                         # Raw & interim data (DVC tracked)
│
├── dvc.yaml                      # DVC pipeline definition (5 stages)
├── params.yaml                   # Hyperparameters config file
├── Dockerfile                    # Container for FastAPI app
├── docker-compose.yml            # Docker compose config
├── .github/workflows/ci-cd.yml   # GitHub Actions CI/CD
│
├── lgbm_model.pkl                # Trained LightGBM model
└── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML PIPELINE (DVC)                        │
│                                                                 │
│  data_ingestion ──► data_preprocessing ──► model_building       │
│        │                  │                      │              │
│   params.yaml        data/raw/            data/interim/         │
│        │                                         │              │
│        └──────────────────────────► model_evaluation            │
│                                           │                     │
│                                    experiment_info.json         │
│                                           │                     │
│                                   model_registration            │
│                                   (MLflow Registry)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                        lgbm_model.pkl
                        tfidf_vectorizer.pkl
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    FASTAPI BACKEND (Docker)                      │
│                                                                  │
│  GET  /                  → Health check                          │
│  POST /predict           → Batch sentiment prediction            │
│  POST /predict_with_timestamps → Sentiment + timestamps          │
│  POST /generate_chart    → Pie chart PNG (StreamingResponse)     │
│  POST /generate_wordcloud→ Word cloud PNG                        │
│  POST /generate_trend_graph→ Monthly trend PNG                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP (localhost:8000)
┌──────────────────────────────▼──────────────────────────────────┐
│               CHROME EXTENSION (Manifest V3)                     │
│                                                                  │
│  popup.js → YouTube Data API v3                                  │
│          → FastAPI /predict_with_timestamps                      │
│          → FastAPI /generate_chart                               │
│          → FastAPI /generate_wordcloud                           │
│          → FastAPI /generate_trend_graph                         │
│  popup.html → Renders charts, metrics, comment list             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Technical Explanation — Key Functions & Code

---

### 5.1 Data Ingestion — `src/data/data_ingestion.py`

#### Purpose
Downloads a labeled Reddit sentiment CSV from GitHub (used as training proxy for comment sentiment) and splits it 80/20 into train/test sets.

#### Key Function: `load_data()`
```python
def load_data(data_url: str) -> pd.DataFrame:
    df = pd.read_csv(data_url)
    return df
```
- **Why**: Centralised, logged data loading. CSV is fetched from a public GitHub URL — no local setup needed.

#### Key Function: `preprocess_data()`
```python
def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['clean_comment'].str.strip() != '']
    return df
```
- **Why**: Removes nulls, exact duplicate rows, and empty comment strings before any splitting occurs.

#### Key Function: `save_data()`
```python
train_data, test_data = train_test_split(final_df, test_size=0.20, random_state=42)
train_data.to_csv('data/raw/train.csv')
test_data.to_csv('data/raw/test.csv')
```
- **Why**: Fixes random seed for reproducibility. `test_size=0.20` is stored in `params.yaml` and tracked by DVC.

---

### 5.2 Data Preprocessing — `src/data/data_preprocessing.py`

#### Purpose
Applies NLP text normalization to raw comment strings.

#### Key Function: `preprocess_comment()`
```python
def preprocess_comment(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
    return comment
```

| Step | What it does | Why |
|------|-------------|-----|
| `.lower()` | Lowercase all text | Normalize case ("Great" = "great") |
| `re.sub(newline)` | Remove `\n` | Multi-line comments break tokenization |
| `re.sub(non-alphanum)` | Remove emojis, special chars | Keep only ML-readable tokens |
| Stopword removal | Remove common filler words | Reduces noise; keep negations like "not", "no" |
| Lemmatization | Reduce words to root form | "running" → "run", "better" → "good" |

> **Critical Design Choice:** Negation words (`not`, `but`, `however`, `no`, `yet`) are **retained** even though they're standard stopwords. This is essential for sentiment — "not good" ≠ "good".

---

### 5.3 Model Building — `src/model/model_building.py`

#### Purpose
Converts cleaned text into TF-IDF features and trains a LightGBM classifier.

#### Key Function: `apply_tfidf()`
```python
def apply_tfidf(train_data, max_features, ngram_range):
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
    return X_train_tfidf, y_train
```

- **`max_features=1000`**: Top 1000 most important terms. Limits dimensionality, prevents overfitting.
- **`ngram_range=(1,3)`**: Captures unigrams, bigrams, trigrams. "not good" and "very helpful" are separate features.
- **Saved as `.pkl`**: Same vectorizer must be used at inference time.

#### Key Function: `train_lgbm()`
```python
def train_lgbm(X_train, y_train, learning_rate, max_depth, n_estimators):
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        is_unbalance=True,
        class_weight='balanced',
        reg_alpha=0.1,   # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        learning_rate=0.09,
        max_depth=20,
        n_estimators=367
    )
    model.fit(X_train, y_train)
    return model
```

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `objective='multiclass'` | 3-class | Positive / Neutral / Negative |
| `is_unbalance=True` | — | Dataset has class imbalance |
| `class_weight='balanced'` | — | Penalizes errors on minority class more |
| `reg_alpha=0.1` | L1 | Sparsity, feature selection |
| `reg_lambda=0.1` | L2 | Weight shrinkage, generalization |
| `learning_rate=0.09` | — | Slow learning; from `params.yaml` |
| `n_estimators=367` | — | Tuned number of trees |

---

### 5.4 Model Evaluation — `src/model/model_evaluation.py`

#### Purpose
Evaluates the model on the test set and tracks everything in MLflow.

#### Key Function: `evaluate_model()`
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm
```

#### MLflow Logging
```python
mlflow.set_experiment('dvc-pipeline-runs')
with mlflow.start_run() as run:
    mlflow.log_param(key, value)            # Hyperparameters
    mlflow.log_metrics({...})               # Precision / Recall / F1
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.sklearn.log_model(model, "lgbm_model", signature=signature)
    save_model_info(run.info.run_id, model_path, 'experiment_info.json')
```

- **`infer_signature()`**: Auto-detects input/output schema → prevents schema drift in production
- **`experiment_info.json`**: Saves `run_id` and model path so `register_model.py` can locate this exact run

---

### 5.5 Model Registration — `src/model/register_model.py`

#### Purpose
Promotes the trained model to the MLflow Model Registry under a named model and transitions it to "Staging".

```python
model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
model_version = mlflow.register_model(model_uri, "yt_chrome_plugin_model")
client.transition_model_version_stage(
    name="yt_chrome_plugin_model",
    version=model_version.version,
    stage="Staging"
)
```

- **Why Model Registry?** Provides versioning, stage transitions (Staging → Production), and audit trail
- **Why Staging?** A human or automated test can validate before promoting to Production

---

### 5.6 DVC Pipeline — `dvc.yaml`

#### Purpose
Defines the entire ML pipeline as a **Directed Acyclic Graph (DAG)** of stages that DVC tracks for reproducibility.

```yaml
stages:
  data_ingestion:      → outputs: data/raw/
  data_preprocessing:  → outputs: data/interim/
  model_building:      → outputs: lgbm_model.pkl, tfidf_vectorizer.pkl
  model_evaluation:    → outputs: experiment_info.json
  model_registration:  → (no file output, registers to MLflow)
```

**Run entire pipeline:**
```bash
dvc repro
```

**Why DVC?**
- Only re-runs stages whose inputs or parameters have changed
- Tracks data files in remote storage (S3) without storing them in Git
- `params.yaml` changes trigger the right stages automatically

---

### 5.7 FastAPI Backend — `app/main.py`

#### App Initialization
```python
app = FastAPI(title="Sentiment API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
- **CORS**: Required so the Chrome extension (different origin) can call the API

#### Request Models (Pydantic)
```python
class CommentItem(BaseModel):
    text: str
    timestamp: str

class CommentsRequest(BaseModel):
    comments: List[str]
```
- **Pydantic BaseModel**: Automatic validation, type checking, and clear API schema

#### Endpoint: `/predict_with_timestamps`
```python
@app.post("/predict_with_timestamps")
def predict_with_timestamps(req: TimestampRequest):
    processed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(processed).toarray()
    predictions = model.predict(transformed).tolist()
    return [{"comment": c, "sentiment": s, "timestamp": t} ...]
```
- Applies same `preprocess_comment()` as training — critical for consistent feature space
- `vectorizer.transform()` (not `fit_transform`) — uses the already fitted vocabulary

#### Endpoint: `/generate_chart`
```python
@app.post("/generate_chart")
def generate_chart(req: SentimentCountRequest):
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    img = io.BytesIO()
    plt.savefig(img, format="PNG", transparent=True)
    img.seek(0)
    return StreamingResponse(img, media_type="image/png")
```
- **`io.BytesIO()`**: Saves image in memory (no disk I/O) → fast API response
- **`StreamingResponse`**: Streams PNG bytes directly to client

#### Endpoint: `/generate_trend_graph`
```python
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
monthly = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
percentages = (monthly.T / totals).T * 100
```
- **`resample("M")`**: Groups data by calendar month
- **Percentage normalization**: Converts raw counts to % so months with different comment volumes are comparable

---

### 5.8 Chrome Extension — `popup.js`

#### Flow
```javascript
// Step 1: Get current YouTube video ID from URL
const videoId = url.match(/youtube\.com\/watch\?v=([\w-]{11})/)[1];

// Step 2: Fetch comments from YouTube Data API v3
const comments = await fetchCommentsFromAPI(videoId);

// Step 3: Send to FastAPI
const predictions = await getSentimentPredictions(comments);

// Step 4: Parallel image fetching
await Promise.all([
    fetchAndSetImage('/generate_chart', { sentiment_counts }, 'distChart'),
    fetchAndSetImage('/generate_trend_graph', { sentiment_data }, 'trendChart'),
    fetchAndSetImage('/generate_wordcloud', { comments }, 'wordcloudImg')
]);
```

#### Key Design: `Promise.all()`
```javascript
await Promise.all([chart, trend, wordcloud]);
```
- Fires all 3 image requests **concurrently** instead of sequentially
- Reduces total wait time from ~3s to ~1s (parallelism)

#### Sentiment Score Normalization
```javascript
const avgSentimentRaw = totalSentimentScore / totalComments; // -1 to 1
const normalizedScore = (((avgSentimentRaw + 1) / 2) * 10).toFixed(1); // 0 to 10
```
- Maps [-1, 1] → [0, 10] for user-friendly display

---

### 5.9 CI/CD Pipeline — `.github/workflows/ci-cd.yml`

```yaml
jobs:
  build-and-test:     # Runs on every push + PR
    - Checkout code
    - Setup Python 3.11
    - pip install -r requirements.txt
    - python -m py_compile *.py   # Syntax validation

  build-and-push:     # Only on push to main
    - Set up Docker Buildx
    - Login to DockerHub
    - Build & push: kartik87580/yt-analysis:latest
```

- **Gate**: `build-and-push` only runs after `build-and-test` passes (`needs: build-and-test`)
- **Secrets**: `DOCKER_USERNAME` and `DOCKER_PASSWORD` stored in GitHub Secrets

---

## 6. Interview Questions & Answers

---

### 🔵 SECTION A: Project Understanding

---

**Q1. Can you walk me through the end-to-end pipeline of your project?**

> **A:** The project has two parts: an ML pipeline and a serving pipeline.
>
> **ML Pipeline (DVC-managed):**
> 1. `data_ingestion.py` downloads a Reddit comment CSV, removes nulls/duplicates, splits 80/20, saves to `data/raw/`
> 2. `data_preprocessing.py` applies NLP normalization (lowercase, remove special chars, remove stopwords while keeping negations, lemmatize) and saves to `data/interim/`
> 3. `model_building.py` fits a TF-IDF vectorizer (1000 features, unigrams-to-trigrams) and trains LightGBM (multiclass, 367 trees, balanced class weights)
> 4. `model_evaluation.py` runs inference on test data, logs metrics and confusion matrix to MLflow, and saves `experiment_info.json`
> 5. `register_model.py` reads the JSON, registers the run to MLflow Model Registry as `yt_chrome_plugin_model`, transitions to Staging
>
> **Serving Pipeline:**
> - FastAPI loads the `.pkl` model and vectorizer at startup
> - Chrome extension fetches YouTube comments → calls FastAPI endpoints → displays charts in the popup

---

**Q2. What problem does this project solve, and why is it valuable?**

> **A:** YouTube creators and brands receive thousands of comments per video, making it impossible to manually gauge audience sentiment. This project provides a one-click browser extension that runs directly on YouTube. In seconds, it fetches up to 200 comments via the YouTube Data API, classifies each one as positive/neutral/negative using an ML model, and displays:
> - A pie chart for overall distribution
> - A word cloud for key themes
> - A monthly trend graph to see if sentiment is improving or worsening over time
> - Individual comment labels
>
> The value is that this happens **in-browser without any data export** — it meets the user exactly where they are.

---

**Q3. Why did you use Reddit data to train a model for YouTube comments?**

> **A:** The Reddit dataset has high-quality, labeled sentiment data that generalizes well to comment-style informal text. YouTube-specific labeled datasets with sentiment labels are rare and expensive. Both platforms share similar language patterns — informal language, slang, short text, and opinion expression. The preprocessing normalizes most platform-specific differences. This is a practical trade-off between data quality/availability and domain specificity. In production, fine-tuning on YouTube-specific labeled data would improve accuracy.

---

### 🔵 SECTION B: Machine Learning

---

**Q4. Why did you choose LightGBM over other models like Logistic Regression, SVM, or a Transformer?**

> **A:** We evaluated several models in the notebooks:
> - **Logistic Regression / Naive Bayes** — fast baseline but poor on imbalanced multi-class with TF-IDF
> - **XGBoost** — good accuracy but slower to train and less memory-efficient than LightGBM
> - **Transformers (BERT)** — highest accuracy but far too slow (~10s per batch) for a real-time browser extension; and overkill for a pickled-model endpoint
>
> **LightGBM** was chosen because:
> - Leaf-wise tree growth (vs. level-wise in XGBoost) → faster convergence
> - Native support for class imbalance (`is_unbalance=True`)
> - Inference is sub-millisecond on CPU with a pickled model
> - Excellent F1 on multi-class text classification

---

**Q5. What is TF-IDF and why ngram_range=(1,3)?**

> **A:** TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical vectors by weighting each word by how frequently it appears in a document vs. how rare it is across all documents.
>
> ```
> TF-IDF(t, d) = TF(t, d) × log(N / df(t))
> ```
>
> Where:
> - `TF(t, d)` = count of term `t` in document `d`
> - `N` = total documents
> - `df(t)` = documents containing term `t`
>
> **`ngram_range=(1,3)` captures:**
> - Unigrams: `"good"`, `"bad"`, `"love"`
> - Bigrams: `"not good"`, `"very helpful"`
> - Trigrams: `"this is amazing"`
>
> This is critical for sentiment because `"not good"` has the opposite meaning from `"good"`. With only unigrams, the model loses negation context.

---

**Q6. The dataset has class imbalance — how did you handle it?**

> **A:** Based on the experiment notebooks (5. Handling_imbalance.ipynb), we tested multiple strategies:
> - **Oversampling (Random)** — duplicates minority class samples
> - **SMOTE-ENN** — synthetic minority oversampling + edited nearest neighbour cleaning
> - **ADASYN** — adaptive synthetic sampling
> - **Undersampling** — removes majority class samples
> - **Class weights** — penalizes wrong predictions on minority class more
>
> The final model uses a combination of `is_unbalance=True` and `class_weight='balanced'` in LightGBM. This is preferred over SMOTE for text data because synthetic text generation doesn't produce realistic comment text. The confusion matrices for each strategy are stored in `notebooks/`.

---

**Q7. What is `lemmatization` and why use it instead of `stemming`?**

> **A:** Both reduce words to their root form:
> - **Stemming**: Crude rule-based truncation — `"running"` → `"run"`, but `"better"` → `"bett"` (incorrect)
> - **Lemmatization**: Linguistic dictionary-based — `"running"` → `"run"`, `"better"` → `"good"` (correct)
>
> We use `WordNetLemmatizer` from NLTK. For sentiment analysis, lemmatization is preferred because incorrect stems can destroy the meaning of sentiment words. "Better" being reduced to "bett" makes it unrecognizable — but reducing it to "good" keeps the positive sentiment signal.

---

**Q8. Why do you retain stopwords like 'not', 'but', 'however', 'no', 'yet'?**

> **A:** Standard stopword lists remove these words because they're high-frequency and low-information in tasks like topic modeling. However, for **sentiment analysis**, negation words completely reverse the sentiment polarity:
> - `"good"` → Positive, but `"not good"` → Negative
> - `"I love it"` → Positive, but `"I do not love it"` → Negative
>
> Removing `"not"` would make these two sentences identical in TF-IDF space and cause sentiment reversal errors. This is a deliberate design decision coded into both the training preprocessing and the inference preprocessing in `app/main.py`.

---

**Q9. What metrics did you track in MLflow and why?**

> **A:** For each sentiment class (Positive, Neutral, Negative):
> - **Precision**: Of predicted positives, how many were actually positive? (avoids false alarms)
> - **Recall**: Of actual positives, how many did we catch? (avoids misses)
> - **F1-Score**: Harmonic mean of precision and recall — balanced metric for imbalanced classes
>
> We also log:
> - **Confusion matrix** (as a PNG artifact) — shows specific failure modes (e.g., does model confuse Neutral with Negative?)
> - **All hyperparameters** from `params.yaml` — so every run is fully reproducible
> - **Model signature** (via `infer_signature`) — enforces input/output schema at serving time

---

### 🔵 SECTION C: MLOps & Engineering

---

**Q10. What is DVC and how is it used in this project?**

> **A:** DVC (Data Version Control) is a Git-like tool for versioning data, models, and ML pipelines.
>
> In this project, `dvc.yaml` defines a 5-stage pipeline as a DAG:
> ```
> data_ingestion → data_preprocessing → model_building → model_evaluation → model_registration
> ```
>
> Running `dvc repro` checks the MD5 hash of each stage's inputs (source files, data, params). If nothing changed, the stage is skipped. If `params.yaml` changes (e.g., `learning_rate`), only `model_building` and downstream stages re-run.
>
> **Benefits in this project:**
> - Reproducibility: Any collaborator can run `dvc repro` and get the same model
> - Data tracking: `data/raw/` and `data/interim/` are tracked without being stored in Git
> - Experiment management: Different parameter configs create different pipeline states

---

**Q11. Explain the MLflow lifecycle in this project.**

> **A:** MLflow provides four components used here:
>
> 1. **Tracking** (`mlflow.set_tracking_uri("http://localhost:5000")`): Logs runs to a local MLflow server. Each run records hyperparameters, metrics, artifacts (confusion matrix PNG, vectorizer PKL).
>
> 2. **Model Logging** (`mlflow.sklearn.log_model(..., signature=signature)`): Logs the LightGBM model with input/output schema information. The `infer_signature()` call captures the expected feature shape from TF-IDF output.
>
> 3. **Model Registry** (`mlflow.register_model(model_uri, "yt_chrome_plugin_model")`): Registers the logged model under a named model in the registry for versioning.
>
> 4. **Stage Transition** (`client.transition_model_version_stage(..., stage="Staging")`): Moves model to "Staging" for validation before production.
>
> The `experiment_info.json` file acts as the handoff artifact between evaluation and registration, carrying the `run_id`.

---

**Q12. How is the FastAPI app containerized and deployed?**

> **A:** The `Dockerfile` builds the app:
> ```dockerfile
> FROM python:3.11-slim-bookworm
> RUN apt-get install -y libgomp1   # LightGBM needs OpenMP
> COPY . /app
> RUN pip install -r requirements.txt
> RUN python -m nltk.downloader stopwords wordnet omw-1.4
> EXPOSE 8000
> CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
> ```
>
> **Key choices:**
> - `python:3.11-slim-bookworm` — minimal image, no unnecessary system packages
> - `libgomp1` — LightGBM requires OpenMP for parallelism; missing it causes runtime crash
> - NLTK data downloaded at build time — avoids download at every container start
>
> The GitHub Actions CI/CD pipeline builds this image on every push to `main` and pushes it to DockerHub as `kartik87580/yt-analysis:latest`.

---

**Q13. What is the Chrome Extension Manifest V3 and how does it work here?**

> **A:** Manifest V3 (MV3) is the current Chrome extension API specification. Key changes from V2:
> - Background scripts replaced by Service Workers
> - Stricter CSP (Content Security Policy)
> - `host_permissions` separated from `permissions`
>
> In this project:
> ```json
> "permissions": ["activeTab", "scripting", "tabs"]
> "host_permissions": ["https://www.googleapis.com/*", "http://localhost/*"]
> ```
>
> - **`activeTab`**: Access the URL of the currently active YouTube tab
> - **`tabs`**: Query tab information (URL, ID)
> - **`host_permissions` for googleapis.com**: Make authenticated requests to YouTube Data API v3
> - **`host_permissions` for localhost**: Call the FastAPI backend
>
> The extension uses a **popup** architecture — `popup.html` and `popup.js` run in the popup window when the extension icon is clicked.

---

**Q14. Why does the extension use `Promise.all()` for image fetching?**

> **A:** Without `Promise.all()`, the three image requests would be sequential:
> ```
> chart (1s) → wordcloud (1s) → trend (1s) = 3 seconds total
> ```
>
> With `Promise.all()`, they fire concurrently:
> ```
> chart  ─────────── (1s)
> wordcloud ──────── (1s)    = ~1 second total
> trend  ─────────── (1s)
> ```
>
> This is critical for UX — a browser extension popup should feel snappy. `Promise.all()` rejects if any single request fails, so individual image failures are caught gracefully in the `fetchAndSetImage()` try-catch.

---

**Q15. How does the backend avoid saving charts to disk?**

> **A:** All chart generation endpoints use `io.BytesIO()` — an in-memory binary buffer:
> ```python
> img = io.BytesIO()
> plt.savefig(img, format="PNG", transparent=True)
> img.seek(0)   # Reset buffer pointer to beginning
> return StreamingResponse(img, media_type="image/png")
> ```
>
> `plt.savefig()` normally writes to a file path, but it also accepts any file-like object. `BytesIO` acts like a file in memory. `img.seek(0)` is mandatory — after writing, the pointer is at the end, so we reset it to position 0 before reading/streaming.
>
> **Benefits:**
> - No disk I/O → faster response
> - No file cleanup needed
> - Thread-safe (each request has its own buffer)
> - Works in Docker containers with read-only filesystems

---

**Q16. How does `preprocess_comment()` ensure training-serving consistency?**

> **A:** This is a critical MLOps concern. The exact same function is defined in:
> - `src/data/data_preprocessing.py` (used during training)
> - `app/main.py` (used during inference)
>
> They apply identically the same steps in the same order:
> 1. lowercase → strip → remove newlines → remove non-alphanum
> 2. remove stopwords (retain negations) → lemmatize
>
> **If preprocessing at serving time differed from training**, the TF-IDF vectorizer would receive different token distributions, causing vocabulary mismatches and degraded model performance (train-serve skew).
>
> Best practice would be to externalize this to a shared utility module (`src/utils/text_utils.py`) to have a single source of truth.

---

### 🔵 SECTION D: System Design

---

**Q17. How would you scale this system to handle 10,000 concurrent users?**

> **A:** The current architecture is single-process. To scale:
>
> 1. **Stateless API**: FastAPI is already stateless (model loaded once at startup) → can run multiple instances
> 2. **Container Orchestration**: Deploy multiple FastAPI replicas with Kubernetes or Docker Swarm
> 3. **Load Balancer**: Nginx or AWS ALB distributes traffic across replicas
> 4. **Model Serving**: Replace pickle loading with a dedicated model server (TorchServe, Triton, or BentoML) with request batching
> 5. **Async Endpoints**: Convert FastAPI routes to `async def` + use async-compatible vectorizer
> 6. **Caching**: Cache recently analyzed video IDs (Redis) — same video analyzed within 1 hour returns cached results
> 7. **Chart Generation**: Offload to a queue (Celery + Redis) and return charts via signed URLs

---

**Q18. What are the limitations of the current approach?**

> **A:**
> - **Local inference only**: The Chrome extension requires `localhost:8000` to be running. Not accessible to end users without a deployed backend.
> - **YouTube API quota**: 10,000 units/day free tier. Fetching 200 comments costs ~3 units. At ~3,333 video analyses/day, quota exhausts.
> - **Reddit → YouTube domain shift**: Model trained on Reddit may underperform on YouTube-specific language (short reactions, emojis, multilingual comments).
> - **No emoji support**: The preprocessing strips emojis (e.g., `"❤️ this video"` loses the positive signal).
> - **Single language**: Only English stopwords and lemmatizer. Non-English comments are processed incorrectly.
> - **API key exposure**: `YOUTUBE_API_KEY` is hardcoded in `popup.js` — visible in extension source. Should use backend as proxy.

---

**Q19. How would you improve model accuracy?**

> **A:**
> 1. **Better features**: Use sentence embeddings (Sentence-BERT) instead of TF-IDF for semantic representation
> 2. **Domain-specific fine-tuning**: Fine-tune `cardiffnlp/twitter-roberta-base-sentiment` on YouTube comments
> 3. **More data**: Collect and label a YouTube-specific dataset
> 4. **Emoji features**: Convert emojis to text (e.g., "❤️" → "love") before preprocessing
> 5. **Multi-language support**: Detect language with `langdetect`, then apply language-specific preprocessing
> 6. **Ensemble**: Combine LightGBM (speed) with a small distilled BERT (accuracy) using confidence thresholding

---

**Q20. What is the role of `params.yaml` and why is it important?**

> **A:** `params.yaml` is the **single source of truth** for all tunable hyperparameters:
> ```yaml
> data_ingestion:
>   test_size: 0.20
> model_building:
>   ngram_range: [1, 3]
>   max_features: 1000
>   learning_rate: 0.09
>   max_depth: 20
>   n_estimators: 367
> ```
>
> **Why it matters:**
> - **DVC integration**: When `params.yaml` changes, DVC detects which stages are affected and re-runs only those
> - **MLflow logging**: All params are logged to each run → `mlflow.log_param(key, value)`
> - **No hardcoding**: Hyperparameter changes don't require code changes → safer experimentation
> - **Reproducibility**: Given the same `params.yaml` + data + code, anyone can reproduce the exact model

---

**Q21. Explain the sentiment score normalization in the extension.**

> **A:** Model predictions use three classes: `-1` (negative), `0` (neutral), `1` (positive).
>
> Average sentiment = `sum(all scores) / total_comments` → range [-1, 1]
>
> To make this user-friendly (0–10 scale):
> ```javascript
> // Map [-1, 1] → [0, 2] → [0, 1] → [0, 10]
> normalizedScore = (((avgSentimentRaw + 1) / 2) * 10).toFixed(1)
> ```
>
> - If all comments are negative: avg = -1 → score = 0.0
> - If all comments are neutral: avg = 0 → score = 5.0
> - If all comments are positive: avg = 1 → score = 10.0
>
> This linear mapping is intuitive and interpretable without requiring any ML knowledge.

---

### 🔵 SECTION E: Behavioral / Soft Skills

---

**Q22. What was the hardest technical challenge in this project?**

> **A:** The most challenging aspect was **training-serving consistency**. When the model was first deployed, there was a subtle difference between the stopwords removed during training vs. during inference, causing the model to receive different vocabulary distributions. This was caught by comparing the classification report on test data vs. live comments. The fix was ensuring both the training and inference preprocessing functions were identical, including the same negation word exclusions.

---

**Q23. What would you build next if you had more time?**

> **A:** Several improvements:
> 1. **Backend URL in extension settings** — so users can configure a remote API endpoint
> 2. **YouTube comment reply analysis** — currently only top-level comments are analyzed
> 3. **Historical video comparison** — analyze multiple videos and compare sentiment side by side
> 4. **Auto-alert system** — notify creator when negative sentiment spikes above a threshold
> 5. **Aspect-based sentiment** — not just overall sentiment, but "sound quality is bad, content is great"
> 6. **Export to PDF/CSV** — let creators download the analysis report

---

**Q24. How did you ensure reproducibility of this ML project?**

> **A:** Multiple layers:
> - **DVC** tracks code, data, and parameter dependencies — `dvc repro` regenerates the exact pipeline
> - **`params.yaml`** centralizes all hyperparameters — no magic numbers in code
> - **`random_state=42`** in `train_test_split` — same split every run
> - **MLflow** logs every run's parameters, metrics, artifacts — historical runs are accessible
> - **Docker** ensures the same Python/OS environment — eliminates "works on my machine"
> - **`requirements.txt`** pins exact package versions — no version drift

---

*Generated on 2026-04-15 | Project: YouTube Comment Sentiment Analysis*
