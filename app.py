import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    DATA_DIR,
    DEFAULT_GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    MAX_COSINE_DISTANCE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    generate_answer,
    get_index_count,
    get_indexed_source_count,
    has_relevant_evidence,
    rebuild_index,
    retrieve,
)

load_dotenv()

st.set_page_config(
    page_title="Campus Companion | RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(180deg, #F7F8FC 0%, #F3F5FA 100%); }
    [data-testid="stHeader"] { background: rgba(247,248,252,0.88); }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
    [data-testid="stSidebar"] { background: #10182B; }
    [data-testid="stSidebar"] * { color: #EEF2FF !important; }
    [data-testid="stSidebar"] input { color: #10182B !important; }
    [data-testid="stSidebar"] .stButton button {
        background: #6D72E8; color: white !important; border: 0; border-radius: 10px;
        font-weight: 700; min-height: 2.7rem;
    }
    [data-testid="stSidebar"] .stButton button:hover { background: #8589F2; }
    .hero {
        background: linear-gradient(120deg, #182544 0%, #303D73 58%, #6558C9 100%);
        padding: 1.65rem 1.8rem; border-radius: 22px; color: white;
        box-shadow: 0 14px 35px rgba(34,45,94,.16); margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 2.1rem;
        font-weight: 800; margin: 0 0 .35rem 0; }
    .hero p { color: #E3E8FF; font-size: 1rem; margin: 0; }
    .eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .72rem;
        font-weight: 700; color: #C8CEFF; margin-bottom: .55rem; }
    .metric-card {
        background: white; border: 1px solid #E7EAF2; border-radius: 15px;
        padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(25,35,70,.045);
    }
    .metric-label { color: #727B91; font-size: .78rem; font-weight: 700; text-transform: uppercase; }
    .metric-value { color: #182544; font-family: 'Manrope', sans-serif; font-size: 1.45rem;
        font-weight: 800; margin-top: .15rem; }
    .section-label { color: #66708A; text-transform: uppercase; letter-spacing: .12em;
        font-size: .72rem; font-weight: 800; margin: 1.25rem 0 .5rem 0; }
    .disclaimer {
        background: #FFF8E8; border: 1px solid #F3E2B7; color: #795A18;
        border-radius: 12px; padding: .75rem 1rem; font-size: .86rem; margin: .8rem 0 1rem;
    }
    .stChatMessage { background: white; border: 1px solid #E8EAF2; border-radius: 16px;
        padding: .75rem 1rem; margin-bottom: .65rem; }
    .stChatInput textarea { border-radius: 14px !important; }
    div[data-testid="stExpander"] { background: white; border: 1px solid #E8EAF2; border-radius: 12px; }
    .stButton button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

with st.sidebar:
    st.markdown(
        "<div style='font-family:Manrope;font-size:1.25rem;font-weight:800;color:white'>🎓 Campus Companion</div>"
        "<div style='color:#AAB5D5;font-size:.8rem;margin:.2rem 0 1rem'>University knowledge assistant</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Project settings")
    api_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Your key is used for generation. Keep it private and never commit .env to GitHub.",
    )
    top_k = st.slider("Retrieved passages (top-k)", 2, 8, 4)
    max_distance = st.slider(
        "Relevance cutoff (cosine distance)",
        min_value=0.50,
        max_value=1.20,
        value=float(MAX_COSINE_DISTANCE),
        step=0.01,
        help="A passage farther than this threshold is treated as weak evidence. Tune using evaluation queries.",
    )

    st.markdown("---")
    st.markdown("### Knowledge base")
    if st.button("⚡ Build / refresh index", use_container_width=True):
        with st.spinner("Reading documents, embedding chunks, and updating ChromaDB..."):
            try:
                file_count, chunk_count = rebuild_index()
                st.success(f"Indexed {file_count} files into {chunk_count} chunks.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.caption(f"Embedding model: `{EMBEDDING_MODEL_NAME.split('/')[-1]}`")
    st.caption(f"Generation model: `{os.getenv('GROQ_MODEL', DEFAULT_GROQ_MODEL)}`")
    st.markdown("---")
    st.markdown("### Add documents")
    uploads = st.file_uploader(
        "Upload TXT, Markdown, or text-based PDF files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Scanned PDFs may need OCR before they can be indexed.",
    )
    if st.button("Save uploads & rebuild", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one file.")
        else:
            upload_dir = DATA_DIR / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for uploaded in uploads:
                safe_name = Path(uploaded.name).name.replace("/", "_").replace("\\", "_")
                if not safe_name:
                    continue
                (upload_dir / safe_name).write_bytes(uploaded.getbuffer())
                saved += 1
            with st.spinner("Rebuilding the vector index..."):
                try:
                    file_count, chunk_count = rebuild_index()
                    st.success(f"Saved {saved} files. Index now contains {chunk_count} chunks.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    st.markdown("---")
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Local documents stay on your computer. LLM prompts are sent to Groq when you ask a question.")

source_count = get_indexed_source_count()
chunk_count = get_index_count()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">DEMO UNIVERSITY • AI KNOWLEDGE DESK</div>
      <h1>Ask. Discover. Understand.</h1>
      <p>Get clear answers from university policy documents, with the evidence shown beside every response.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_data = [
    ("SOURCE DOCUMENTS", str(source_count)),
    ("INDEXED CHUNKS", str(chunk_count)),
    ("RETRIEVAL", f"Top {top_k}"),
    ("EVIDENCE CUTOFF", f"{max_distance:.2f}"),
]
for col, (label, value) in zip(metric_cols, metric_data):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='disclaimer'><b>Demo dataset:</b> The included 25 policy documents are simulated examples for this assignment, not official rules of a real university.</div>",
    unsafe_allow_html=True,
)

if chunk_count == 0:
    st.warning("Your vector index is empty. Use **Build / refresh index** in the sidebar before asking questions.")

st.markdown("<div class='section-label'>Suggested questions</div>", unsafe_allow_html=True)
suggestions = [
    "What is the minimum attendance requirement?",
    "How is cumulative GPA calculated?",
    "When can I withdraw from a course?",
    "What are the graduation requirements?",
]
suggestion_cols = st.columns(4)
for col, suggestion in zip(suggestion_cols, suggestions):
    with col:
        if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()

# Render conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔎 Retrieved evidence"):
                for index, item in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**{index}. `{item['source']}` · chunk {item['chunk']} · "
                        f"cosine distance {item['distance']:.3f}**"
                    )
                    st.write(item["text"])

prompt = st.chat_input("Ask a question about university policies…")
if prompt is None and st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = ""

if prompt:
    question = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if chunk_count == 0:
            answer = "The knowledge base is empty. Build the vector index from the sidebar first."
            sources = []
        else:
            try:
                with st.spinner("Searching the policy library and preparing your answer…"):
                    sources = retrieve(question, top_k=top_k)
                    answer = generate_answer(
                        question,
                        sources,
                        api_key=api_key.strip() or None,
                        max_distance=max_distance,
                    )
                st.markdown(answer)
                if sources:
                    with st.expander("🔎 Retrieved evidence", expanded=not has_relevant_evidence(sources, max_distance)):
                        for index, item in enumerate(sources, start=1):
                            st.markdown(
                                f"**{index}. `{item['source']}` · chunk {item['chunk']} · "
                                f"cosine distance {item['distance']:.3f}**"
                            )
                            st.write(item["text"])
            except Exception as exc:
                answer = f"I couldn't complete the request: {exc}"
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

st.markdown(
    "<div style='text-align:center;color:#8A91A5;font-size:.75rem;margin-top:2rem'>"
    "Retrieval-Augmented Generation · Sentence Transformers · ChromaDB · Groq · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
