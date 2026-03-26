import streamlit as st
import os
import time
import hashlib
from io import BytesIO
from datetime import datetime

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG PDF Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #0b0d11;
    --surface:   #13161d;
    --surface2:  #1c2030;
    --border:    #252a38;
    --accent:    #00e5ff;
    --accent2:   #7c3aed;
    --gold:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --danger:    #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 60%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(0,229,255,0.12), rgba(124,58,237,0.12));
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 100px;
    padding: 4px 16px;
    font-size: 0.72rem;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.8rem;
    letter-spacing: 1px;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 12px; margin: 1.2rem 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 100px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 2px;
}

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 14px; margin: 1rem 0; }

.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(124,58,237,0.15));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 75%;
    font-size: 0.92rem;
}
.bubble-user .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(124,58,237,0.9);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 5px;
}

.bubble-ai {
    align-self: flex-start;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 20px;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.65;
}
.bubble-ai .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}

/* ── Confidence bar ── */
.conf-wrap { margin-top: 10px; }
.conf-label {
    font-size: 0.68rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.conf-bar-bg {
    background: var(--border);
    border-radius: 100px;
    height: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 4px;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.6s ease;
}
.conf-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    margin-top: 3px;
}

/* ── Source chunks ── */
.chunk-card {
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.12);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.82rem;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
.chunk-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    color: var(--accent);
    letter-spacing: 2px;
    margin-bottom: 6px;
    text-transform: uppercase;
}

/* ── Divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.3;
    margin: 1.5rem 0;
}

/* ── Input box ── */
[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(124,58,237,0.15)) !important;
    border: 1px solid rgba(0,229,255,0.3) !important;
    border-radius: 10px !important;
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(0,229,255,0.2) !important;
    transform: translateY(-1px);
}

/* ── Sidebar elements ── */
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

/* ── PDF metadata card ── */
.pdf-meta {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 1rem 0;
}
.pdf-meta-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text);
    word-break: break-all;
}
.pdf-meta-row {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.pdf-meta-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
}
.pdf-meta-item span {
    color: var(--gold);
    font-weight: 600;
}

/* ── Toast-like status ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--success);
    letter-spacing: 1px;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Suggested questions ── */
.suggest-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.8rem 0; }
.suggest-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}
.suggest-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Select/multiselect ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
for key, default in [
    ("vector_store", None),
    ("pdf_name", None),
    ("pdf_hash", None),
    ("pdf_pages", 0),
    ("pdf_chunks", 0),
    ("pdf_words", 0),
    ("chat_history", []),
    ("qa_chain", None),
    ("hf_token_ok", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def extract_text_and_meta(pdf_file):
    reader = PdfReader(pdf_file)
    pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    words = len(text.split())
    return text, pages, words


def get_text_chunks(text, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks):
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


def build_qa_chain(vector_store, hf_token):
    """Build a RetrievalQA chain with a proper prompt template."""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are DocMind, an expert document analyst. Answer the question clearly and helpfully using ONLY the context below.
If the answer is not in the context, say: "I couldn't find that in the document."
Be concise, specific, and structured. Use bullet points when listing multiple items.

Context:
{context}

Question: {question}

Answer:""",
    )

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=hf_token,
        temperature=0.3,
        max_new_tokens=512,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.6},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )
    return qa_chain


def compute_confidence(docs, question):
    """Heuristic confidence: avg similarity score approximation via chunk relevance."""
    if not docs:
        return 0.0
    q_words = set(question.lower().split())
    scores = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(q_words & doc_words) / max(len(q_words), 1)
        scores.append(min(overlap * 4.5, 1.0))
    return round(sum(scores) / len(scores) * 100, 1)


def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


def suggest_questions(pdf_name):
    """Auto-generate starter questions based on common document types."""
    name_lower = pdf_name.lower()
    if any(k in name_lower for k in ["report", "annual", "quarter"]):
        return ["What are the key findings?", "What are the main risks mentioned?", "What are the financial highlights?"]
    elif any(k in name_lower for k in ["research", "paper", "study"]):
        return ["What is the main hypothesis?", "What methodology was used?", "What were the conclusions?"]
    elif any(k in name_lower for k in ["contract", "agreement", "terms"]):
        return ["What are the key obligations?", "What are the termination clauses?", "Who are the parties involved?"]
    else:
        return ["What is this document about?", "What are the main topics covered?", "Summarize the key points."]


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
    background:linear-gradient(135deg,#00e5ff,#7c3aed);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px;">
    🧠 DocMind
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#64748b;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:1.5rem;">
    RAG · FAISS · LangChain
    </div>
    """, unsafe_allow_html=True)

    # HuggingFace Token
    st.markdown("### 🔑 HuggingFace Token")
    hf_token = st.text_input(
        "API Token",
        type="password",
        placeholder="hf_...",
        help="Get your free token at huggingface.co/settings/tokens",
        value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
    )
    if hf_token:
        st.session_state["hf_token_ok"] = True
        st.markdown('<div class="status-pill"><div class="status-dot"></div>TOKEN ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.warning("Add your HuggingFace token to enable LLM answers", icon="⚠️")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # PDF Upload
    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type="pdf",
        label_visibility="collapsed",
    )

    # Chunk settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 400, 1500, 800, 50)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 150, 25)
        top_k = st.slider("Sources to Retrieve", 2, 6, 4)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Stats
    if st.session_state["pdf_name"]:
        st.markdown(f"""
        <div class="pdf-meta">
            <div class="pdf-meta-title">📄 {st.session_state['pdf_name']}</div>
            <div class="pdf-meta-row">
                <div class="pdf-meta-item">Pages<br><span>{st.session_state['pdf_pages']}</span></div>
                <div class="pdf-meta-item">Words<br><span>{st.session_state['pdf_words']:,}</span></div>
                <div class="pdf-meta-item">Chunks<br><span>{st.session_state['pdf_chunks']}</span></div>
                <div class="pdf-meta-item">Q&A<br><span>{len(st.session_state['chat_history']) // 2}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat History"):
            st.session_state["chat_history"] = []
            st.rerun()

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem;color:#475569;font-family:'JetBrains Mono',monospace;line-height:1.8;">
    Built by <span style="color:#00e5ff;">Vijay Dokka</span><br>
    <a href="https://github.com/Arawn-D" style="color:#7c3aed;text-decoration:none;">@Arawn-D</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 class="hero-title">DocMind</h1>
    <p class="hero-sub">Retrieval-Augmented Intelligence</p>
    <div class="hero-badge">⚡ FAISS · MiniLM Embeddings · MMR Search · Flan-T5 LLM</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    current_hash = file_hash(file_bytes)

    # Only reprocess if it's a new file
    if current_hash != st.session_state.get("pdf_hash"):
        with st.spinner("🔍 Extracting & indexing your document..."):
            raw_text, pages, words = extract_text_and_meta(BytesIO(file_bytes))

            if not raw_text.strip():
                st.error("❌ Could not extract text. Make sure the PDF isn't scanned/image-only.")
                st.stop()

            chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vector_store = build_vector_store(chunks)

            # Build QA chain if token available
            qa_chain = None
            if hf_token:
                try:
                    qa_chain = build_qa_chain(vector_store, hf_token)
                    st.session_state["qa_chain"] = qa_chain
                    st.session_state["llm_error"] = None
                except Exception as e:
                    st.session_state["llm_error"] = str(e)
                    st.warning(f"⚠️ LLM init failed: {e}")

            st.session_state.update({
                "vector_store": vector_store,
                "pdf_name": uploaded_file.name,
                "pdf_hash": current_hash,
                "pdf_pages": pages,
                "pdf_chunks": len(chunks),
                "pdf_words": words,
                "chat_history": [],
            })

        st.success(f"✅ Indexed **{len(chunks)} chunks** from **{pages} pages** — ready to query!")

# ─────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────
if st.session_state["vector_store"] is not None:

    # Stats row
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">{st.session_state['pdf_pages']}</div>
            <div class="stat-label">Pages</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{st.session_state['pdf_words']:,}</div>
            <div class="stat-label">Words</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{st.session_state['pdf_chunks']}</div>
            <div class="stat-label">Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{len(st.session_state['chat_history']) // 2}</div>
            <div class="stat-label">Questions Asked</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Suggested questions ──
    if not st.session_state["chat_history"]:
        st.markdown("**💡 Suggested questions:**")
        suggestions = suggest_questions(st.session_state["pdf_name"])
        cols = st.columns(len(suggestions))
        for i, (col, q) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(q, key=f"suggest_{i}"):
                    st.session_state["_prefill"] = q
                    st.rerun()

    # ── Render chat history ──
    if st.session_state["chat_history"]:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="bubble-user">
                    <div class="label">You</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                conf = msg.get("confidence", 0)
                conf_color = "#10b981" if conf > 60 else "#f59e0b" if conf > 35 else "#ef4444"
                # Format answer: convert newlines and bullet points to HTML
                raw = msg['content']
                formatted = (raw
                    .replace("&", "&amp;")
                    .replace("<em", "[[EM]]").replace("</em>", "[[/EM]]")
                    .replace("<", "&lt;").replace(">", "&gt;")
                    .replace("[[EM]]", "<em").replace("[[/EM]]", "</em>")
                    .replace("\n• ", "<br>• ")
                    .replace("\n- ", "<br>• ")
                    .replace("\n* ", "<br>• ")
                    .replace("\n\n", "<br><br>")
                    .replace("\n", "<br>")
                )
                st.markdown(f"""
                <div class="bubble-ai">
                    <div class="label">🧠 DocMind</div>
                    <div style="line-height:1.7;">{formatted}</div>
                    <div class="conf-wrap">
                        <div class="conf-label">RELEVANCE CONFIDENCE</div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{conf}%;background:linear-gradient(90deg,{conf_color},{conf_color}88);"></div>
                        </div>
                        <div class="conf-pct" style="color:{conf_color};">{conf}%</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Source chunks in expander
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source passages"):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div class="chunk-header">Source Chunk {i+1}</div>
                                {src[:400]}{'...' if len(src) > 400 else ''}
                            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Input ──
    prefill = st.session_state.pop("_prefill", "")

    with st.form(key="chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask anything about your document",
            value=prefill,
            placeholder="e.g. What are the main conclusions?",
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("⚡ Ask", use_container_width=False)

    if ask_clicked and user_q.strip():
        question = user_q.strip()

        with st.spinner("🧠 Thinking..."):
            docs = st.session_state["vector_store"].similarity_search(question, k=top_k)
            sources = [d.page_content for d in docs]
            confidence = compute_confidence(docs, question)

            # Try LLM answer first, fallback to best chunk
            answer = None
            if st.session_state.get("qa_chain"):
                try:
                    result = st.session_state["qa_chain"].invoke({"query": question})
                    answer = result.get("result", "").strip()
                    if not answer or answer.lower().startswith("i couldn"):
                        answer = None
                except Exception as e:
                    answer = f"<em style='color:#ef4444;font-size:0.8rem;'>⚠️ LLM error: {e}</em>"

            if not answer:
                err = st.session_state.get("llm_error")
                if err:
                    fallback_label = f"<em style='color:#ef4444;font-size:0.8rem;'>⚠️ LLM failed: {err[:120]}...</em><br><br>"
                else:
                    fallback_label = "<em style='color:#64748b;font-size:0.8rem;'>[Retrieval-only — add HF token for LLM answers]</em><br><br>"
                answer = fallback_label + (docs[0].page_content if docs else "No relevant content found.")

        # Store in history
        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "sources": sources,
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        st.rerun()

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;color:#475569;">
        <div style="font-size:4rem;margin-bottom:1rem;">📄</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#64748b;">
            Upload a PDF to begin
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.85rem;margin-top:0.5rem;color:#334155;">
            Your document stays local — embeddings built in-browser, nothing sent to the cloud
        </div>
        <div style="display:flex;justify-content:center;gap:2rem;margin-top:2.5rem;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:1.5rem;">🔍</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">Semantic Search</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.5rem;">⚡</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">MMR Retrieval</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.5rem;">🧠</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">LLM Answers</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.5rem;">📎</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">Source Tracing</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
