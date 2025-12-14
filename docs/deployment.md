# 🚀 Deploying to Streamlit Community Cloud

This guide explains how to deploy the **Document Knowledge Retrieval Tool** to [Streamlit Community Cloud](https://streamlit.io/cloud) securely and for free.

## ✅ Prerequisites

1. **GitHub Account**: Your project code must be in a public (or private) GitHub repository.
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
3. **API Keys**: You will need your `OPENAI_API_KEY`, `MILVUS_URI`, and `MILVUS_TOKEN`.

---

## 🛠️ Step-by-Step Deployment Guide

### 1. Push Your Code to GitHub
Ensure your latest code is pushed to your GitHub repository.
- Make sure `requirements.txt` is present in the root directory (this tells Streamlit what to install).
- **IMPORTANT**: Do NOT commit your `.env` file containing real API keys. Use `.gitignore` to keep it local.

### 2. Connect to Streamlit Cloud
1. Go to your [Streamlit Dashboard](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your repository, branch (usually `main` or `master`), and the main file path (`app.py`).

### 3. Configure Secrets (CRITICAL) 🔐
Streamlit Cloud uses a "Secrets" management system to handle environment variables securely. **Do not skip this step.**

1. Before clicking "Deploy", click on **"Advanced settings"**.
2. Go to the **"Secrets"** section.
3. Paste the contents of your `.env` file here, in TOML format:

```toml
OPENAI_API_KEY = "sk-..."
MILVUS_URI = "https://in03-..."
MILVUS_TOKEN = "db_admin:..."
# Optional configuration
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
```

> **Note**: These values are encrypted and injected into your app as environment variables at runtime. safely.

### 4. Deploy 🚀
1. Click **"Save"** on the secrets dialog.
2. Click **"Deploy!"**.
3. Streamlit will start building your app. You can watch the logs to see dependencies installing.

---

## 🔄 Updating Your App
Streamlit Cloud automatically detects pushes to your repository. When you push new code to GitHub, your app will automatically re-deploy with the changes!

## ⚠️ Troubleshooting
- **Module not found**: Ensure the missing package is listed in `requirements.txt`.
- **Secrets error**: Double-check your secrets in the Streamlit dashboard settings. They must be in valid TOML format (e.g., `KEY = "VALUE"`).
