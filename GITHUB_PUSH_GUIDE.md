# Publish the project to GitHub

Run these commands in PowerShell from inside the extracted `Campus-Companion-RAG` folder.

```powershell
git init
git add .
git status
git commit -m "Build Campus Companion RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Campus-Companion-RAG.git
git push -u origin main
```

Before pushing, check that `.env` is not staged:

```powershell
git status
```

The `.gitignore` file excludes `.env`, `.venv/`, `chroma_db/`, and Python bytecode. Never commit an API key. Create an empty GitHub repository first, then replace the remote URL above with your repository URL.
