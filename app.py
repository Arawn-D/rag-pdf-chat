import os
import hashlib
from io import BytesIO
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate



# Page Config
st.set_page_config(
    page_title="DocMind - RAG PDF Chat",
    page_icon="🧠",


# Custom CSS with SVG Icons
DOC_MIND_CSS = r"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap");

:root {
  --bg: #0b0d11;
  --surface: #13161d;
  --surface2: #1c2030;
  --border: #252a38;
  --accent: #00e5ff;
  --accent2: #7c3aed;
  --gold: #f59e0b;
  --text: #e2e8f0;
  --muted: #64748b;
  --success: #10b981;
  --danger: #ef4444;
}
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Inter", sans-serif;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  position: relative;
}
.hero-title {
  font-family: "Syne", sans-serif;
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
  font-family: "JetBrains Mono", monospace;
  font-size: 0.78rem;
  color: var(--muted);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-top: 0.4rem;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(90deg, rgba(0,229,255,0.12), rgba(124,58,237,0.12));
  border: 1px solid rgba(0,229,255,0.25);
  border-radius: 100px;
  padding: 4px 16px;
  font-size: 0.72rem;
  color: var(--accent);
  font-family: "JetBrains Mono", monospace;
  margin-top: 0.8rem;
  letter-spacing: 1px;
}
.hero-badge svg {
  width: 14px;
  height: 14px;
  fill: var(--accent);
}
.stat-row {
  display: flex;
  gap: 12px;
  margin: 1.2rem 0;
  flex-wrap: wrap;
}
.stat-card {
  flex: 1;
  min-width: 120px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
  text-align: center;
}
.stat-val {
  font-family: "Syne", sans-serif;
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
.chat-wrap {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin: 1rem 0;
}
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
  font-family: "JetBrains Mono", monospace;
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
  font-family: "JetBrains Mono", monospace;
  font-size: 0.65rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 6px;
}
.conf-wrap { margin-top: 10px; }
.conf-label {
  font-size: 0.68rem;
  color: var(--muted);
  font-family: "JetBrains Mono", monospace;
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
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  color: var(--accent);
  margin-top: 3px;
}
.chunk-card {
  background: rgba(0,229,255,0.04);
  border: 1px solid rgba(0,229,255,0.12);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 10px 14px;
  margin-top: 8px;
  font-size: 0.82rem;
  color: #94a3b8;
  font-family: "Inter", sans-serif;
  line-height: 1.6;
}
.chunk-header {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.63rem;
  color: var(--accent);
  letter-spacing: 2px;
  margin-bottom: 6px;
  text-transform: uppercase;
}
.glow-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.3;
  margin: 1.5rem 0;
}
[data-testid="stTextInput"] input {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: "Inter", sans-serif !important;
  padding: 12px 16px !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
}
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(124,58,237,0.15)) !important;
  border: 1px solid rgba(0,229,255,0.3) !important;
  border-radius: 10px !important;
  color: var(--accent) !important;
  font-family: "Syne", sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.5px !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  background: rgba(0,229,255,0.2) !important;
  transform: translateY(-1px);
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: "Syne", sans-serif !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] label {
  color: var(--muted) !important;
  font-size: 0.78rem !important;
  text-transform: uppercase !important;
  letter-spacing: 1.5px !important;
}
.pdf-meta {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-top: 2px solid var(--accent);
  border-radius: 12px;
  padding: 16px 20px;
  margin: 1rem 0;
}
.pdf-meta-title {
  font-family: "Syne", sans-serif;
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
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--muted);
}
.pdf-meta-item span {
  color: var(--gold);
  font-weight: 600;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(16,185,129,0.12);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: 100px;
  padding: 4px 14px;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  color: var(--success);
  letter-spacing: 1px;
}
.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--success);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.suggest-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0.8rem 0;
}
.suggest-chip {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 100px;
  padding: 5px 14px;
  font-size: 0.78rem;
  color: var(--muted);
  cursor: pointer;
  transition: all 0.2s;
  font-family: "Inter", sans-serif;
}
.suggest-chip:hover {
  border-color: var(--accent);
  color: var(--accent);
}
[data-testid="stExpander"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
  color: var(--muted) !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.75rem !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
[data-testid="stSpinner"] { color: var(--accent) !important; }
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
"""



# Session State
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
    ("llm_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# SVG Icon Definitions
SVG_BRAIN = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5c.67 0 1.35.09 2 .26 1.78-2 5.03-2.84 6.42-2.26 1.4.58-.42 7-.42 7 .57 1.07 1 2.24 1 3.44C21 17.9 16.97 21 12 21S3 17.9 3 13.44C3 12.24 3.43 11.07 4 10c0 0-1.82-6.42-.42-7 1.39-.58 4.64.26 6.42 2.26 .65-.17 1.33-.26 2-.26z"/></svg>'''
SVG_PDF = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'''
SVG_TOKEN = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="14" x2="15" y2="14"/><circle cx="16" cy="16" r="2"/></svg>'''
SVG_SEARCH = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>'''
SVG_MMR = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>'''
SVG_LINK = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>'''
SVG_SETTINGS = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'''
SVG_TRASH = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'''
SVG_PAPER = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>'''
SVG_PAGES = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>'''
SVG_CHUNKS = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>'''
SVG_QUESTIONS = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><circle cx="12" cy="12" r="10"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'''
SVG_LIGHTBULB = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6a2 2 0 0 1 2 2v1a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2v-1a2 2 0 0 1 2-2z"/><path d="M12 2a7 7 0 0 0-7 7c0 2 1 4 3 5V9a4 4 0 0 0 8 0v5c2-1 3-3 3-5a7 7 0 0 0-7-7z"/></svg>'''
SVG_SEND = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>'''
SVG_DOC = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>'''


# Helper Functions
def extract_text_and_meta(pdf_file: BytesIO):
    reader = PdfReader(pdf_file)
    pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    words = len(text.split())
    return text, pages, words


def get_text_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 150):
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


def build_qa_chain(vector_store, hf_token: str):
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are DocMind, an expert document analyst. Answer the question "
            "clearly and helpfully using ONLY the context below.\n"
            "If the answer is not in the context, say: I could not find that in the document.\n"
            "Be concise, specific, and structured. Use bullet points for lists.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    )
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=hf_token,
        temperature=0.3,
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


def compute_confidence(docs, question: str) -> float:
    if not docs:
        return 0.0
    q_words = set(question.lower().split())
    scores = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(q_words & doc_words) / max(len(q_words), 1)
        scores.append(min(overlap * 4.5, 1.0))
    return round(sum(scores) / len(scores) * 100, 1)


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def suggest_questions(pdf_name: str) -> list:
    name_lower = pdf_name.lower()
    if any(k in name_lower for k in ["report", "annual", "quarter"]):
        return ["What are the key findings?", "What are the main risks mentioned?", "What are the financial highlights?"]
    elif any(k in name_lower for k in ["research", "paper", "study"]):
        return ["What is the main hypothesis?", "What methodology was used?", "What were the conclusions?"]
    elif any(k in name_lower for k in ["contract", "agreement", "terms"]):
        return ["What are the key obligations?", "What are the termination clauses?", "Who are the parties involved?"]
    else:
        return ["What is this document about?", "What are the main topics covered?", "Summarize the key points."]


def format_answer_html(raw: str) -> str:
    if raw.startswith("<em") or raw.startswith("<span"):
        return raw
    formatted = (
        raw
        .replace("&", "&amp;")
        .replace("\n• ", "<br>• ")
        .replace("\n- ", "<br>• ")
        .replace("\n* ", "<br>• ")
        .replace("\n\n", "<br><br>")
        .replace("\n", "<br>")
    )
    return formatted


# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
    background:linear-gradient(135deg,#00e5ff,#7c3aed);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px;">
    {SVG_BRAIN} DocMind
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#64748b;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:1.5rem;">
    RAG &middot; FAISS &middot; LangChain
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ")
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;color:var(--text);margin-bottom:8px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">{SVG_TOKEN} HuggingFace Token</div>""", unsafe_allow_html=True)
    hf_token = st.text_input(
        "",
        type="password",
        placeholder="hf_...",
        help="Get your free token at huggingface.co/settings/tokens",
        value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
        label_visibility="collapsed",
    )
    if hf_token:
        st.session_state["hf_token_ok"] = True
        st.markdown('<div class="status-pill"><div class="status-dot"></div>TOKEN ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.warning("Add your HuggingFace token to enable LLM answers", icon="⚠️")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ")
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;color:var(--text);margin-bottom:8px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">{SVG_PDF} Upload PDF</div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type="pdf",
        label_visibility="collapsed",
    )

    with st.expander(f"{SVG_SETTINGS} Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 400, 1500, 800, 50)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 150, 25)
        top_k = st.slider("Sources to Retrieve", 2, 6, 4)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    if st.session_state["pdf_name"]:
        st.markdown(f"""
        <div class="pdf-meta">
        <div class="pdf-meta-title">{SVG_DOC} {st.session_state['pdf_name']}</div>
        <div class="pdf-meta-row">
        <div class="pdf-meta-item">Pages<br><span>{st.session_state['pdf_pages']}</span></div>
        <div class="pdf-meta-item">Words<br><span>{st.session_state['pdf_words']:,}</span></div>
        <div class="pdf-meta-item">Chunks<br><span>{st.session_state['pdf_chunks']}</span></div>
        <div class="pdf-meta-item">Q&A<br><span>{len(st.session_state['chat_history']) // 2}</span></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["chat_history"]:
            if st.button(f"{SVG_TRASH} Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.7rem;color:#475569;font-family:'JetBrains Mono',monospace;line-height:1.8;">
    Built by <span style="color:#00e5ff;">Vijay Dokka</span><br>
    <a href="https://github.com/Arawn-D" style="color:#7c3aed;text-decoration:none;">@Arawn-D</a>
    </div>


# Main Content
st.markdown(f"""
<div class="hero">
<h1 class="hero-title">DocMind</h1>
<p class="hero-sub">Retrieval-Augmented Intelligence</p>
<div class="hero-badge">
{SVG_PAPER} FAISS
{SVG_LINK} MiniLM
{SVG_MMR} MMR
{SVG_BRAIN} Flan-T5
</div>
</div>
""", unsafe_allow_html=True)

# PDF Processing
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    current_hash = file_hash(file_bytes)
    if current_hash != st.session_state.get("pdf_hash"):
        with st.spinner("Extracting and indexing your document..."):
            raw_text, pages, words = extract_text_and_meta(BytesIO(file_bytes))
            if not raw_text.strip():
                st.error("Could not extract text. Make sure the PDF is not scanned/image-only.")
                st.stop()
            chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vector_store = build_vector_store(chunks)
            qa_chain = None
            if hf_token:
                try:
                    qa_chain = build_qa_chain(vector_store, hf_token)
                    st.session_state["qa_chain"] = qa_chain
                    st.session_state["llm_error"] = None
                except Exception as e:
                    st.session_state["llm_error"] = str(e)
                    st.warning(f"LLM init failed: {e}")
            st.session_state.update({
                "vector_store": vector_store,
                "pdf_name": uploaded_file.name,
                "pdf_hash": current_hash,
                "pdf_pages": pages,
                "pdf_chunks": len(chunks),
                "pdf_words": words,
                "chat_history": [],
            })


# Chat Interface
if st.session_state["vector_store"] is not None:
    # Stats Row
    st.markdown(f"""
    <div class="stat-row">
    <div class="stat-card">
    <div class="stat-val">{SVG_PAGES} {st.session_state['pdf_pages']}</div>
    <div class="stat-label">Pages</div>
    </div>
    <div class="stat-card">
    <div class="stat-val">{st.session_state['pdf_words']:,}</div>
    <div class="stat-label">Words</div>
    </div>
    <div class="stat-card">
    <div class="stat-val">{SVG_CHUNKS} {st.session_state['pdf_chunks']}</div>
    <div class="stat-label">Chunks</div>
    </div>
    <div class="stat-card">
    <div class="stat-val">{SVG_QUESTIONS} {len(st.session_state['chat_history']) // 2}</div>
    <div class="stat-label">Questions Asked</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Suggested Questions
    if not st.session_state["chat_history"]:
        st.markdown(f"**{SVG_LIGHTBULB} Suggested questions:**")
        suggestions = suggest_questions(st.session_state["pdf_name"])
        cols = st.columns(len(suggestions))
        for i, (col, q) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(q, key=f"suggest_{i}"):
                    st.session_state["_prefill"] = q
                    st.rerun()

    # Render Chat History
    if st.session_state["chat_history"]:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="bubble-user">
                <div class="label">You</div>
                {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                conf = msg.get("confidence", 0)
                conf_color = "#10b981" if conf > 60 else "#f59e0b" if conf > 35 else "#ef4444"
                raw = msg['content']
                formatted = format_answer_html(raw)
                st.markdown(f"""
                <div class="bubble-ai">
                <div class="label">{SVG_BRAIN} DocMind</div>
                <div style="line-height:1.7;">{formatted}</div>
                <div class="conf-wrap">
                <div class="conf-label">RELEVANCE CONFIDENCE</div>
                <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{conf}%;background:linear-gradient(90deg,{conf_color},{conf_color}88);"></div>
                </div>
                <div class="conf-pct" style="color:{conf_color};">{conf}%</div>
                </div>
                </div>
                """, unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander(f"{SVG_LINK} {len(msg['sources'])} source passages"):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <div class="chunk-card">
                            <div class="chunk-header">Source Chunk {i+1}</div>
                            {src[:400]}{'...' if len(src) > 400 else ''}
                            </div>
                            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input
    prefill = st.session_state.pop("_prefill", "")
    with st.form(key="chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "",
            value=prefill,
            placeholder="Ask anything about your document",
            label_visibility="collapsed",
        )
        if st.form_submit_button(f"{SVG_SEND} Ask", use_container_width=False):
            if user_q.strip():
                question = user_q.strip()
                with st.spinner("Thinking..."):
                    docs = st.session_state["vector_store"].similarity_search(question, k=top_k)
                    sources = [d.page_content for d in docs]
                    confidence = compute_confidence(docs, question)
                    answer = None
                    if st.session_state.get("qa_chain"):
                        try:
                            result = st.session_state["qa_chain"].invoke({"query": question})
                            answer = result.get("result", "").strip()
                            if not answer or answer.lower().startswith("i could"):
                                answer = None
                        except Exception as e:
                            answer = f"<em style='color:#ef4444;font-size:0.8rem;'>LLM error: {e}</em>"
                    if not answer:
                        err = st.session_state.get("llm_error")
                        if err:
                            fallback_label = f"<em style='color:#ef4444;font-size:0.8rem;'>LLM failed: {err[:120]}...<br><br>" + (docs[0].page_content if docs else "No relevant content found.") + "</em>"
                        else:
                            fallback_label = f"<em style='color:#64748b;font-size:0.8rem;'>[Retrieval-only — add HF token for LLM answers]<br><br></em>" + (docs[0].page_content if docs else "No relevant content found.")
                        answer = fallback_label
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
    # Empty State
    st.markdown(f"""
    <div style="text-align:center;padding:5rem 2rem;color:#475569;">
    <div style="font-size:4rem;margin-bottom:1rem;">{SVG_PDF}</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#64748b;">
    Upload a PDF to begin
    </div>
    <div style="font-family:'Inter',sans-serif;font-size:0.85rem;margin-top:0.5rem;color:#334155;">
    Your document stays local — embeddings built in-browser, nothing sent to the cloud
    </div>
    <div style="display:flex;justify-content:center;gap:2rem;margin-top:2.5rem;flex-wrap:wrap;">
    <div style="text-align:center;">
    <div style="font-size:1.5rem;color:#475569;">{SVG_SEARCH}</div>
    <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">Semantic Search</div>
    </div>
    <div style="text-align:center;">
    <div style="font-size:1.5rem;color:#475569;">{SVG_MMR}</div>
    <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">MMR Retrieval</div>
    </div>
    <div style="text-align:center;">
    <div style="font-size:1.5rem;color:#475569;">{SVG_BRAIN}</div>
    <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">LLM Answers</div>
    </div>
    <div style="text-align:center;">
    <div style="font-size:1.5rem;color:#475569;">{SVG_LINK}</div>
    <div style="font-size:0.75rem;color:#475569;margin-top:4px;font-family:'JetBrains Mono',monospace;">Source Tracing</div>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)        st.success(f"Indexed **{len(chunks)} chunks** from **{pages} pages** — ready to query!")    """, unsafe_allow_html=True)st.markdown(DOC_MIND_CSS, unsafe_allow_html=True)    layout="wide",
    initial_sidebar_state="expanded",
)load_dotenv()
