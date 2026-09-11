---
title: Automated Fake News Detection
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Automated Fake News Detection System

Web demo for the GCTU final-year project: classify English news text as **Likely Real** or **Likely Fake**, with a confidence score and **LIME** explanation (highlighted tokens + top-10 feature chart).

> **Caveat:** This is an automated statistical assessment, not a verified fact-check.

**Default model:** Linear SVM — chosen for optimal performance and fast LIME explanations on CPU-only hosting. SVM achieves 99.35% accuracy on the ISOT dataset while providing significantly faster inference and explanation generation compared to BERT. Trained artifacts for BERT, BiLSTM, and Naive Bayes are included under `results/` for research purposes.

---

## Deploy to Hugging Face Spaces (recommended)

### Step 1 — Create the Space

1. Sign in at [huggingface.co](https://huggingface.co).
2. Open [huggingface.co/new-space](https://huggingface.co/new-space).
3. **Space name:** e.g. `fake-news-detection-gctu`
4. **SDK:** **Docker** (not Gradio/Streamlit)
5. **Hardware:** **CPU basic** (16 GB RAM — enough for all bundled models)
6. Create the Space.

### Step 2 — Prepare files on your PC

From your project folder (`run`), you need these paths in the Space repository:

```
Dockerfile
README.md                 ← this file (with the YAML block at the top)
requirements-deploy.txt
.dockerignore
.gitattributes
templates/
static/
results/                  ← must include model weights (see below)
Automated Fake News Detection/app/
```

**Model weights (required):** your local `results/` folder should contain at least:

| Folder | Files |
|--------|--------|
| `results/svm/` | `model.joblib`, `vectorizer.joblib`, `metadata.json` |
| `results/bert/` | `model.safetensors`, `config.json`, tokenizer files, `metadata.json` |
| `results/bilstm/` | `model.pt`, `vocab.json`, `metadata.json` |
| `results/naive_bayes/` | `model.joblib`, `vectorizer.joblib`, `metadata.json` |

These are gitignored in your main project repo but **must be copied into the Space**.

### Step 3 — Upload via Git (preferred for large files)

Install [Git LFS](https://git-lfs.com/) once on your machine, then:

```powershell
# Clone the empty Space (replace YOUR_USERNAME and SPACE_NAME)
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME hf-space
cd hf-space

# Copy deployment files from the project
$SRC = "C:\Users\coura\Desktop\L400\final year project\run"
Copy-Item "$SRC\Dockerfile" .
Copy-Item "$SRC\requirements-deploy.txt" .
Copy-Item "$SRC\README.md" .
Copy-Item "$SRC\.dockerignore" .
Copy-Item "$SRC\.gitattributes" .
Copy-Item "$SRC\templates" "templates" -Recurse
Copy-Item "$SRC\static" "static" -Recurse
Copy-Item "$SRC\results" "results" -Recurse
New-Item -ItemType Directory -Force -Path "Automated Fake News Detection" | Out-Null
Copy-Item "$SRC\Automated Fake News Detection\app" "Automated Fake News Detection\app" -Recurse
```

Then LFS and push:

```powershell
git lfs install
git lfs track "*.safetensors" "*.pt" "*.joblib"
git add .
git commit -m "Add Docker deployment and trained models"
git push
```

First push will ask for your Hugging Face credentials (access token with write access).

The Space builds automatically; open **Logs** if the build fails.

### Step 3 (alternative) — Upload via the website

1. Open your Space → **Files** tab.
2. Upload `Dockerfile`, `requirements-deploy.txt`, this `README.md`, `.dockerignore`.
3. Create folders and upload `templates/`, `static/`, `results/`, and `Automated Fake News Detection/app/` (drag folders or upload zip).

Large files (`model.safetensors`, ~438 MB) may require **Git LFS** or the HF web UI’s large-file upload — Git is more reliable for BERT.

### Step 4 — Test the live URL

After the build succeeds (usually 5–15 minutes):

`https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

Submit a news article (25+ words) and confirm label, confidence, and LIME chart appear.

---

## Test the Docker image locally (optional)

From the project root, with `results/` populated:

```powershell
cd "C:\Users\coura\Desktop\L400\final year project\run"
docker build -t fake-news-detector .
docker run --rm -p 7860:7860 fake-news-detector
```

Open [http://localhost:7860](http://localhost:7860).

To serve BERT instead of SVM locally (slow LIME):

```powershell
docker run --rm -p 7860:7860 -e DEMO_MODEL=bert fake-news-detector
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_FACTCHECK_API_KEY` | unset | Google Cloud API key for the Fact Check Tools API. When set, the result page shows a "External fact-checks" section listing published reviews (Snopes, PolitiFact, AFP, etc.) matching the article's central claim. When unset the section is hidden — classification is unaffected. Create a free key: console.cloud.google.com → enable "Fact Check Tools API" → Credentials → API key. |
| `DEMO_MODEL` | Not used | Legacy variable - SVM is now the default for production use |

---

## Team

Courage Bosomtwe, Michael Kwesi Degah, Bashar Saeed Adams — Ghana Communication Technology University.
