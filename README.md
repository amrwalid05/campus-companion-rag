# Campus Companion — University Policy RAG Assistant

A polished, end-to-end Retrieval-Augmented Generation (RAG) application for asking questions about a small library of university-policy documents.

> **Dataset disclaimer:** The 25 included policy files are fictional/simulated examples created for this assignment. They are not official rules of any real university. Replace them with instructor-approved documents if required.

## What the project demonstrates

1. **Ingestion:** loads TXT, Markdown, and text-based PDF documents.
2. **Chunking:** creates sentence-aware chunks with overlap (default 520 characters / 90-character overlap).
3. **Embeddings:** uses a multilingual Sentence Transformers model locally.
4. **Vector storage:** persists vectors and metadata in ChromaDB using cosine distance.
5. **Retrieval:** embeds the question and retrieves the top-k relevant passages.
6. **Augmentation:** combines the user question with retrieved passages and their filenames.
7. **Generation:** calls Groq-hosted `openai/gpt-oss-20b` with a strict grounding prompt.
8. **Interface:** provides a Streamlit chat UI with evidence expanders, source counts, upload controls, and a relevance cutoff.
9. **Evaluation:** includes retrieval-side Hit@4 / unsupported-query checks and a manual evaluation template.

## Project structure

```text
Campus-Companion-RAG/
├── app.py                    # Streamlit chat application
├── rag_engine.py             # Full RAG pipeline
├── ingest.py                 # Build / refresh vector index
├── evaluate.py               # Retrieval evaluation script
├── requirements.txt          # Dependencies
├── .env.example               # Safe environment template
├── .gitignore
├── README.md
├── START_HERE.txt
├── report.pdf                 # 7-page technical report
├── test_questions.csv         # 10 test questions + expected evidence
├── evaluation_template.csv    # Manual relevance / faithfulness / correctness log
├── data/                      # 25 simulated policy documents
├── tests/
│   └── test_chunking.py
└── .streamlit/config.toml      # UI theme
```

## Requirements

- **Recommended for this package:** Python 3.14.x, 64-bit (the setup below uses the `py` launcher).
- Windows 10/11, macOS, or Linux.
- Internet connection for installing Python packages and downloading the embedding model the first time.
- A Groq API key for answer generation.
- No OpenAI API key is required.

ChromaDB 1.5.3 added Python 3.14 support in its Python client, and Sentence Transformers 6.3.0 lists Python 3.14 support. Keep dependencies current if installing on Python 3.14.

## Windows setup (PowerShell)

Open PowerShell **inside the project folder**. Check that Python is available:

```powershell
py -3.14 --version
```

Create the virtual environment:

```powershell
py -3.14 -m venv .venv
```

Install the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create the private `.env` file and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

Paste your Groq key after `GROQ_API_KEY=` and save. Then build the index:

```powershell
.\.venv\Scripts\python.exe ingest.py
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit normally opens at `http://localhost:8501`.

### If the virtual environment does not appear

Run these checks from the project directory:

```powershell
py -0p
py -3.14 --version
Get-ChildItem .venv\Scripts
```

If `py -3.14` is not detected, rerun the Python installer and enable **Add python.exe to PATH** and the Python Launcher option. Open a new PowerShell window afterward.

## macOS / Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Groq key
python ingest.py
streamlit run app.py
```

## Create a Groq API key

1. Sign in to GroqCloud.
2. Open the API Keys page:
3. Create a key and copy it.
4. Put it in the local `.env` file after `GROQ_API_KEY=`.
5. Never send the key in chat or upload `.env` to GitHub.

Default generation model: `openai/gpt-oss-20b`. Check Groq's current supported-model list if the provider changes availability: https://console.groq.com/docs/models

## How to use the app

1. Run `ingest.py` once, or click **Build / refresh index** in the sidebar.
2. Ask a question using the chat box or one of the suggested prompts.
3. Open **Retrieved evidence** under the answer to inspect the text, filename, chunk number, and cosine distance.
4. Upload your own TXT, Markdown, or text-based PDF documents from the sidebar and rebuild the index.
5. Tune the cosine-distance cutoff using `test_questions.csv` and `evaluate.py`.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | empty | Secret key used only for generation |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model ID |
| `EMBEDDING_MODEL_NAME` | multilingual MiniLM | Local embedding model |
| `CHUNK_SIZE` | `520` | Maximum chunk length in characters |
| `CHUNK_OVERLAP` | `90` | Approximate overlap between chunks |
| `MAX_COSINE_DISTANCE` | `0.82` | Initial relevance/abstention cutoff; calibrate with test data |

The cutoff is a heuristic starting point, not a universal threshold. Review the observed distances in evaluation and tune it for the selected embedding model and dataset.

## Evaluation

Run retrieval-side evaluation after indexing:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

The script reports:
- **Source Hit@4:** whether the expected source appears among the top four retrieved chunks for supported questions.
- **Unsupported-query abstention:** whether a question with no expected source is rejected by the configured cosine-distance cutoff.

Then use `evaluation_template.csv` to score answer generation manually:
- **Relevance (0–2):** 0 = irrelevant, 1 = partially relevant, 2 = directly relevant evidence.
- **Faithfulness (0–2):** 0 = unsupported claims, 1 = partly supported, 2 = all factual claims supported by retrieved text.
- **Correctness (0–2):** 0 = incorrect, 1 = partially correct, 2 = correct answer.

Do not claim measured scores until you run the application and fill in the evaluation log. `test_questions.csv` contains expected evidence and answer guidance, not fabricated model outputs.

## Unit tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Security and limitations

- `.env` is excluded by `.gitignore`; never commit API keys.
- The embedding model runs locally; the question and retrieved context are sent to Groq for answer generation.
- Text-based PDFs are supported. Scanned/image-only PDFs need OCR before ingestion.
- The system can still fail if documents are incomplete or retrieval selects weak evidence. The cosine-distance cutoff reduces, but does not eliminate, this risk.
- Simulated policies are for demonstration only and must not be used as real university guidance.

## GitHub submission checklist

Include code, `data/`, `requirements.txt`, `.env.example`, `README.md`, `report.pdf`, test files, and genuine screenshots from your running application. Use `GITHUB_PUSH_GUIDE.md` for publishing. Do **not** upload `.env`, `.venv/`, `chroma_db/`, API keys, or private data.
