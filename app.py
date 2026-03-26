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

load_dotenv()

st.set_page_config(
    page_title="DocMind",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◈</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

ICON = {
    "logo":    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>',
    "key":     '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="15" r="5"/><path d="M21 2l-9.6 9.6"/><path d="M15.5 7.5l3 3L22 7l-3-3"/></svg>',
    "upload":  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "settings":'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>',
    "trash":   '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>',
    "send":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>',
    "doc":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "source":  '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>',
    "brain":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9.5 2a2.5 2.5 0 010 5H9a7 7 0 000 14h6a7 7 0 000-14h-.5a2.5 2.5 0 010-5H15"/><path d="M9 9h6M9 13h6M9 17h6"/></svg>',
    "spark":   '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "summary": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
    "check":   '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:#080a0f; --bg2:#0d1017; --surface:#111318; --surface2:#161a24; --surface3:#1e2333;
  --border:#1f2537; --border2:#2a3045;
  --accent:#6366f1; --accent-lt:#818cf8; --accent-glow:rgba(99,102,241,0.15);
  --teal:#14b8a6; --teal-glow:rgba(20,184,166,0.12);
  --gold:#f59e0b; --text:#e2e8f0; --text2:#94a3b8; --text3:#475569;
  --success:#10b981; --danger:#f43f5e;
  --r:14px; --rs:8px; --rl:20px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"]>div:first-child{padding-top:1.5rem;}
[data-testid="stSidebarNav"]{display:none!important;}
.main .block-container{padding:2rem 2.5rem!important;max-width:880px!important;}
::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:10px;}

/* SIDEBAR */
.sb-logo{display:flex;align-items:center;gap:10px;padding:0 1.2rem 1.4rem;border-bottom:1px solid var(--border);margin-bottom:1.4rem;}
.sb-logo-mark{width:36px;height:36px;background:linear-gradient(135deg,var(--accent),var(--teal));border-radius:10px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px var(--accent-glow);}
.sb-logo-mark svg{color:white;}
.sb-logo-text{font-size:1.05rem;font-weight:700;color:var(--text);letter-spacing:-0.3px;}
.sb-logo-ver{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--text3);letter-spacing:1px;text-transform:uppercase;}
.sb-lbl{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--text3);text-transform:uppercase;letter-spacing:2px;margin-bottom:7px;display:flex;align-items:center;gap:5px;padding:0 1.2rem;}

/* Inputs */
[data-testid="stTextInput"] input{background:var(--surface2)!important;border:1px solid var(--border2)!important;border-radius:var(--rs)!important;color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-size:0.82rem!important;padding:10px 14px!important;transition:border-color .2s,box-shadow .2s!important;caret-color:var(--accent)!important;}
[data-testid="stTextInput"] input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-glow)!important;outline:none!important;}
[data-testid="stTextInput"] input::placeholder{color:var(--text3)!important;}
[data-testid="stTextInput"] label{display:none!important;}
.token-ok{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:100px;padding:5px 12px;font-family:'JetBrains Mono',monospace;font-size:0.63rem;color:var(--success);letter-spacing:1.5px;text-transform:uppercase;margin-top:7px;}
.token-dot{width:5px;height:5px;border-radius:50%;background:var(--success);animation:blink 2.5s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
.token-warn{font-size:0.73rem;color:var(--gold);margin-top:6px;padding:8px 10px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.18);border-radius:6px;}

/* Upload */
[data-testid="stFileUploader"]{background:var(--surface2)!important;border:1.5px dashed var(--border2)!important;border-radius:var(--r)!important;transition:border-color .2s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;}
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] section{color:var(--text2)!important;font-size:0.8rem!important;}

/* Slider */
[data-testid="stSlider"]>div>div{background:var(--border2)!important;}
[data-testid="stSlider"] [data-testid="stSliderThumb"]{background:var(--accent)!important;box-shadow:0 0 8px var(--accent-glow)!important;}
[data-testid="stSlider"] label{color:var(--text2)!important;font-size:0.76rem!important;}

/* Expander */
[data-testid="stExpander"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:var(--rs)!important;}
[data-testid="stExpander"] summary{color:var(--text2)!important;font-size:0.78rem!important;font-weight:500!important;padding:10px 14px!important;}

/* Buttons */
.stButton>button{background:transparent!important;border:1px solid var(--border2)!important;border-radius:var(--rs)!important;color:var(--text2)!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-size:0.8rem!important;font-weight:500!important;padding:7px 12px!important;transition:all .2s!important;width:100%!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent-lt)!important;background:var(--accent-glow)!important;}
[data-testid="stFormSubmitButton"]>button{background:linear-gradient(135deg,var(--accent),#4f46e5)!important;border:none!important;border-radius:var(--rs)!important;color:white!important;font-weight:600!important;font-size:0.84rem!important;padding:10px 22px!important;box-shadow:0 4px 16px rgba(99,102,241,0.35)!important;transition:all .2s!important;width:auto!important;letter-spacing:.2px!important;}
[data-testid="stFormSubmitButton"]>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 24px rgba(99,102,241,0.45)!important;}

/* PDF card */
.pdf-card{background:linear-gradient(135deg,var(--surface2),var(--surface3));border:1px solid var(--border2);border-top:2px solid var(--teal);border-radius:var(--r);padding:13px 15px;margin:1rem 0;animation:fadeIn .4s ease;}
.pdf-card-name{font-size:0.8rem;font-weight:600;color:var(--text);word-break:break-all;display:flex;align-items:flex-start;gap:5px;margin-bottom:9px;}
.pdf-card-name svg{flex-shrink:0;margin-top:2px;color:var(--teal);}
.pdf-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;}
.pdf-stat{background:var(--surface);border-radius:6px;padding:5px 7px;text-align:center;}
.pdf-stat-val{font-family:'JetBrains Mono',monospace;font-size:0.88rem;font-weight:600;color:var(--teal);}
.pdf-stat-lbl{font-size:0.56rem;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-top:1px;}

/* PAGE HEADER */
.page-hdr{display:flex;align-items:center;gap:14px;margin-bottom:1.8rem;padding-bottom:1.4rem;border-bottom:1px solid var(--border);}
.page-hdr-icon{width:46px;height:46px;background:linear-gradient(135deg,var(--accent),var(--teal));border-radius:13px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 28px var(--accent-glow),0 0 28px var(--teal-glow);flex-shrink:0;}
.page-hdr-icon svg{width:20px;height:20px;color:white;}
.page-hdr-title{font-size:1.55rem;font-weight:700;color:var(--text);letter-spacing:-.5px;line-height:1.2;}
.page-hdr-sub{font-size:0.8rem;color:var(--text3);margin-top:2px;}
.page-hdr-badges{margin-left:auto;display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end;}
.badge{background:var(--surface2);border:1px solid var(--border2);border-radius:100px;padding:3px 9px;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--text3);letter-spacing:1px;text-transform:uppercase;}

/* STATS */
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:1.4rem;}
.stat-tile{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:15px;position:relative;overflow:hidden;transition:border-color .2s,transform .2s;}
.stat-tile:hover{border-color:var(--border2);transform:translateY(-1px);}
.stat-tile::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--teal));opacity:0;transition:opacity .2s;}
.stat-tile:hover::before{opacity:1;}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:700;color:var(--text);line-height:1;}
.stat-lbl{font-size:0.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;}
.stat-ico{position:absolute;top:13px;right:13px;opacity:0.12;color:var(--accent);}

/* SUMMARY */
.summary-card{background:linear-gradient(135deg,rgba(99,102,241,0.06),rgba(20,184,166,0.06));border:1px solid rgba(99,102,241,0.18);border-radius:var(--r);padding:18px 22px;margin-bottom:1.4rem;animation:slideUp .5s ease;}
.summary-hdr{display:flex;align-items:center;gap:7px;margin-bottom:10px;}
.summary-ttl{font-size:0.68rem;font-weight:600;color:var(--accent-lt);text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;}
.summary-body{font-size:0.88rem;color:var(--text2);line-height:1.75;}

/* DIVIDER */
.div{height:1px;background:var(--border);margin:1.4rem 0;}

/* SUGGEST */
.sug-lbl{font-size:0.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:9px;display:flex;align-items:center;gap:5px;}

/* CHAT */
.chat-wrap{display:flex;flex-direction:column;gap:12px;padding:4px 0;}
.row-user{display:flex;justify-content:flex-end;}
.row-ai{display:flex;justify-content:flex-start;align-items:flex-start;gap:10px;}
.ai-av{width:30px;height:30px;background:linear-gradient(135deg,var(--accent),var(--teal));border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;}
.ai-av svg{width:13px;height:13px;color:white;}
.bub-meta{font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:var(--text3);margin-bottom:6px;letter-spacing:1px;text-transform:uppercase;}
.bub-user{background:linear-gradient(135deg,rgba(99,102,241,0.18),rgba(99,102,241,0.08));border:1px solid rgba(99,102,241,0.22);border-radius:16px 16px 4px 16px;padding:12px 15px;max-width:72%;font-size:0.87rem;color:var(--text);line-height:1.6;animation:slideL .3s ease;}
.bub-ai{background:var(--surface2);border:1px solid var(--border2);border-radius:4px 16px 16px 16px;padding:14px 17px;max-width:84%;font-size:0.87rem;color:var(--text);line-height:1.75;animation:slideR .3s ease;}
.conf-row{display:flex;align-items:center;gap:7px;margin-top:11px;padding-top:9px;border-top:1px solid var(--border);}
.conf-track{flex:1;height:3px;background:var(--border);border-radius:10px;overflow:hidden;}
.conf-fill{height:3px;border-radius:10px;transition:width .8s cubic-bezier(.4,0,.2,1);}
.conf-pct{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--text3);white-space:nowrap;min-width:30px;text-align:right;}
.conf-lbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--text3);text-transform:uppercase;letter-spacing:1px;white-space:nowrap;}
.rtag{display:inline-flex;align-items:center;gap:4px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.18);border-radius:4px;padding:2px 7px;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--gold);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px;}
.src-chip{background:rgba(20,184,166,0.05);border:1px solid rgba(20,184,166,0.13);border-left:2px solid var(--teal);border-radius:6px;padding:9px 12px;margin-top:6px;font-size:0.79rem;color:var(--text2);line-height:1.65;}
.src-hdr{font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:var(--teal);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;display:flex;align-items:center;gap:4px;}
.llm-err{background:rgba(244,63,94,0.06);border:1px solid rgba(244,63,94,0.18);border-radius:6px;padding:8px 12px;font-size:0.76rem;color:var(--danger);font-family:'JetBrains Mono',monospace;margin-bottom:10px;}
.proc-banner{background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(20,184,166,0.08));border:1px solid rgba(99,102,241,0.18);border-radius:var(--r);padding:14px 18px;display:flex;align-items:center;gap:10px;margin-bottom:1rem;animation:fadeIn .4s ease;}
.proc-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:blink 1.5s ease-in-out infinite;}
.proc-text{font-size:0.83rem;color:var(--text2);}

/* FORM */
[data-testid="stForm"]{background:var(--surface2)!important;border:1px solid var(--border2)!important;border-radius:var(--r)!important;padding:5px 5px 5px 14px!important;transition:border-color .2s,box-shadow .2s!important;}
[data-testid="stForm"]:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-glow)!important;}
[data-testid="stForm"] [data-testid="stTextInput"] input{background:transparent!important;border:none!important;box-shadow:none!important;padding:10px 0!important;font-size:0.9rem!important;font-family:'Plus Jakarta Sans',sans-serif!important;}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus{box-shadow:none!important;}
[data-testid="stHorizontalBlock"]{align-items:center!important;gap:6px!important;}

/* EMPTY */
.empty{text-align:center;padding:4.5rem 2rem;}
.empty-icon{width:68px;height:68px;background:linear-gradient(135deg,var(--surface2),var(--surface3));border:1px solid var(--border2);border-radius:18px;display:flex;align-items:center;justify-content:center;margin:0 auto 1.4rem;}
.empty-icon svg{width:28px;height:28px;color:var(--text3);}
.empty-title{font-size:1.15rem;font-weight:600;color:var(--text2);margin-bottom:7px;}
.empty-sub{font-size:0.83rem;color:var(--text3);max-width:370px;margin:0 auto 2.2rem;line-height:1.65;}
.feat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;max-width:400px;margin:0 auto;}
.feat-tile{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:13px 15px;text-align:left;transition:border-color .2s;}
.feat-tile:hover{border-color:var(--accent);}
.feat-ico{width:26px;height:26px;background:var(--accent-glow);border-radius:6px;display:flex;align-items:center;justify-content:center;margin-bottom:7px;}
.feat-ico svg{width:12px;height:12px;color:var(--accent-lt);}
.feat-name{font-size:0.78rem;font-weight:600;color:var(--text2);}
.feat-desc{font-size:0.7rem;color:var(--text3);margin-top:2px;}

/* ANIMATIONS */
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideL{from{opacity:0;transform:translateX(8px)}to{opacity:1;transform:translateX(0)}}
@keyframes slideR{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}

/* Spinner */
[data-testid="stSpinner"] p{color:var(--text2)!important;font-size:0.8rem!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────
for k, v in {
    "vector_store": None, "pdf_name": None, "pdf_hash": None,
    "pdf_pages": 0, "pdf_chunks": 0, "pdf_words": 0,
    "chat_history": [], "qa_chain": None,
    "llm_error": None, "auto_summary": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────
def extract_pdf(f):
    r = PdfReader(f); pages = len(r.pages); text = ""
    for p in r.pages:
        t = p.extract_text()
        if t: text += t + "\n"
    return text, pages, len(text.split())

def chunk_text(text, size=800, overlap=150):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_text(text)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_store(chunks):
    return FAISS.from_texts(chunks, embedding=get_embeddings())

def build_chain(store, token):
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=token,
        temperature=0.2, max_new_tokens=512,
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are DocMind, a precise document analysis assistant.\n"
            "Answer the question using ONLY the context provided below.\n"
            "Be specific, clear, and well-structured. Use bullet points for lists.\n"
            "If the answer is not in the context, say: Not found in document.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    )
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 12, "lambda_mult": 0.65},
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

def auto_summarise(store, token):
    try:
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            huggingfacehub_api_token=token,
            temperature=0.1, max_new_tokens=300,
        )
        docs = store.similarity_search("main topics summary key points overview", k=5)
        ctx  = "\n\n".join(d.page_content for d in docs)
        return llm.invoke(
            "You are a document analyst. Write a concise 3-4 sentence professional "
            "summary covering the main topics and key points from the context below.\n\n"
            f"Context:\n{ctx}\n\nSummary:"
        ).strip()
    except Exception:
        return None

def conf_score(docs, q):
    if not docs: return 0.0
    qw = set(q.lower().split())
    return round(sum(min(len(qw & set(d.page_content.lower().split())) / max(len(qw),1) * 4.5, 1.0) for d in docs) / len(docs) * 100, 1)

def md5(b): return hashlib.md5(b).hexdigest()

def suggest(name):
    n = name.lower()
    if any(k in n for k in ["report","annual","quarter","financial"]):
        return ["What are the key findings?","What risks are mentioned?","What are the recommendations?"]
    if any(k in n for k in ["research","paper","study","thesis"]):
        return ["What is the main hypothesis?","What methodology was used?","What are the conclusions?"]
    if any(k in n for k in ["contract","agreement","terms","legal"]):
        return ["Who are the parties involved?","What are the key obligations?","What are the termination clauses?"]
    if any(k in n for k in ["resume","cv","curriculum"]):
        return ["What are this person's skills?","What is their work experience?","What projects have they built?"]
    return ["Summarize this document","What are the main topics?","What are the key points?"]

def to_html(raw):
    if not raw: return ""
    return (raw
        .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        .replace("\n- ","<br>&bull; ").replace("\n* ","<br>&bull; ")
        .replace("\n- ","<br>&bull; ").replace("\n\n","<br><br>").replace("\n","<br>")
    )

# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo">
      <div class="sb-logo-mark">{ICON['logo']}</div>
      <div>
        <div class="sb-logo-text">DocMind</div>
        <div class="sb-logo-ver">v2.0 &nbsp;&middot;&nbsp; RAG Engine</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-lbl">{ICON["key"]} HuggingFace Token</div>', unsafe_allow_html=True)
    hf_token = st.text_input(
        "token", type="password", placeholder="hf_xxxxxxxxxxxxxxxx",
        value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""), label_visibility="collapsed",
    )
    if hf_token:
        st.markdown('<div class="token-ok"><div class="token-dot"></div>Authenticated</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="token-warn">Add token to enable LLM answers</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:var(--border);margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sb-lbl">{ICON["upload"]} Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type="pdf", label_visibility="collapsed")

    st.markdown('<div style="margin-top:.8rem;"></div>', unsafe_allow_html=True)
    with st.expander("Advanced Settings"):
        chunk_size    = st.slider("Chunk size", 400, 1500, 800, 50)
        chunk_overlap = st.slider("Chunk overlap", 50, 400, 150, 25)
        top_k         = st.slider("Sources per answer", 2, 8, 5)

    st.markdown('<div style="height:1px;background:var(--border);margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    if st.session_state["pdf_name"]:
        st.markdown(f"""
        <div class="pdf-card">
          <div class="pdf-card-name">{ICON['doc']} {st.session_state['pdf_name']}</div>
          <div class="pdf-stats">
            <div class="pdf-stat"><div class="pdf-stat-val">{st.session_state['pdf_pages']}</div><div class="pdf-stat-lbl">Pages</div></div>
            <div class="pdf-stat"><div class="pdf-stat-val">{st.session_state['pdf_words']:,}</div><div class="pdf-stat-lbl">Words</div></div>
            <div class="pdf-stat"><div class="pdf-stat-val">{st.session_state['pdf_chunks']}</div><div class="pdf-stat-lbl">Chunks</div></div>
            <div class="pdf-stat"><div class="pdf-stat-val">{len(st.session_state['chat_history'])//2}</div><div class="pdf-stat-lbl">Q&amp;A</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state["chat_history"]:
            if st.button("Clear conversation"):
                st.session_state["chat_history"] = []
                st.rerun()

    st.markdown("""
    <div style="position:fixed;bottom:1rem;left:0;width:260px;padding:0 1.2rem;
    font-size:0.65rem;color:#334155;font-family:'JetBrains Mono',monospace;line-height:1.9;">
      Built by <span style="color:#818cf8;">Vijay Dokka</span>
      &nbsp;&middot;&nbsp;
      <a href="https://github.com/Arawn-D" style="color:#14b8a6;text-decoration:none;">@Arawn-D</a>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-hdr">
  <div class="page-hdr-icon">{ICON['logo']}</div>
  <div>
    <div class="page-hdr-title">DocMind</div>
    <div class="page-hdr-sub">Intelligent document analysis &amp; Q&amp;A</div>
  </div>
  <div class="page-hdr-badges">
    <span class="badge">FAISS</span>
    <span class="badge">MiniLM</span>
    <span class="badge">MMR</span>
    <span class="badge">Flan-T5</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── PDF PROCESSING ──
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    cur_hash   = md5(file_bytes)

    if cur_hash != st.session_state["pdf_hash"]:
        with st.spinner("Extracting text and building index..."):
            raw, pages, words = extract_pdf(BytesIO(file_bytes))
            if not raw.strip():
                st.error("Could not extract text. The PDF may be scanned or image-based.")
                st.stop()

            chunks = chunk_text(raw, chunk_size, chunk_overlap)
            store  = build_store(chunks)
            qa_chain, llm_err = None, None

            if hf_token:
                try:
                    qa_chain = build_chain(store, hf_token)
                except Exception as e:
                    llm_err = str(e)

            summary = None
            if qa_chain:
                with st.spinner("Generating document summary..."):
                    summary = auto_summarise(store, hf_token)

            st.session_state.update({
                "vector_store": store, "pdf_name": uploaded_file.name,
                "pdf_hash": cur_hash, "pdf_pages": pages,
                "pdf_chunks": len(chunks), "pdf_words": words,
                "chat_history": [], "qa_chain": qa_chain,
                "llm_error": llm_err, "auto_summary": summary,
            })

        st.markdown(f"""
        <div class="proc-banner">
          <div class="proc-dot"></div>
          <div class="proc-text">
            Indexed <strong>{len(chunks)} chunks</strong> from
            <strong>{pages} page{"s" if pages!=1 else ""}</strong> &mdash; ready to query
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── CHAT UI ──
if st.session_state["vector_store"] is not None:

    qa_count = len(st.session_state["chat_history"]) // 2
    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-tile">
        <div class="stat-ico">{ICON['doc']}</div>
        <div class="stat-num">{st.session_state['pdf_pages']}</div>
        <div class="stat-lbl">Pages</div>
      </div>
      <div class="stat-tile">
        <div class="stat-num">{st.session_state['pdf_words']:,}</div>
        <div class="stat-lbl">Words</div>
      </div>
      <div class="stat-tile">
        <div class="stat-num">{st.session_state['pdf_chunks']}</div>
        <div class="stat-lbl">Chunks</div>
      </div>
      <div class="stat-tile">
        <div class="stat-ico">{ICON['brain']}</div>
        <div class="stat-num">{qa_count}</div>
        <div class="stat-lbl">Questions</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["llm_error"]:
        st.markdown(f'<div class="llm-err">{ICON["spark"]} LLM error: {st.session_state["llm_error"][:200]}</div>', unsafe_allow_html=True)

    if st.session_state["auto_summary"]:
        st.markdown(f"""
        <div class="summary-card">
          <div class="summary-hdr">{ICON['summary']}<span class="summary-ttl">Document Summary</span></div>
          <div class="summary-body">{to_html(st.session_state['auto_summary'])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    if not st.session_state["chat_history"]:
        suggestions = suggest(st.session_state["pdf_name"])
        st.markdown(f'<div class="sug-lbl">{ICON["spark"]} Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(len(suggestions))
        for i, (col, q) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(q, key=f"sq_{i}"):
                    st.session_state["_prefill"] = q
                    st.rerun()
        st.markdown('<div style="margin-bottom:.8rem;"></div>', unsafe_allow_html=True)

    if st.session_state["chat_history"]:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="row-user">
                  <div>
                    <div class="bub-meta" style="text-align:right;">You &nbsp;&middot;&nbsp; {msg.get('ts','')}</div>
                    <div class="bub-user">{msg['content']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                cv    = msg.get("confidence", 0)
                cc    = "#10b981" if cv > 65 else "#f59e0b" if cv > 35 else "#f43f5e"
                rtag  = '<div class="rtag">Retrieval-only &mdash; add HF token for LLM answers</div>' if msg.get("retrieval_only") else ""
                st.markdown(f"""
                <div class="row-ai">
                  <div class="ai-av">{ICON['brain']}</div>
                  <div>
                    <div class="bub-meta">DocMind &nbsp;&middot;&nbsp; {msg.get('ts','')}</div>
                    <div class="bub-ai">
                      {rtag}
                      <div>{to_html(msg['content'])}</div>
                      <div class="conf-row">
                        <span class="conf-lbl">Relevance</span>
                        <div class="conf-track"><div class="conf-fill" style="width:{cv}%;background:{cc};"></div></div>
                        <span class="conf-pct" style="color:{cc};">{cv}%</span>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander(f"View {len(msg['sources'])} source passages"):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <div class="src-chip">
                              <div class="src-hdr">{ICON['source']} Source {i+1}</div>
                              {src[:450]}{'...' if len(src)>450 else ''}
                            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    prefill = st.session_state.pop("_prefill", "")
    with st.form(key="qf", clear_on_submit=True):
        c1, c2 = st.columns([8, 1])
        with c1:
            user_q = st.text_input(
                "q", value=prefill,
                placeholder="Ask anything about your document...",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("Ask")

    if submitted and user_q.strip():
        q  = user_q.strip()
        ts = datetime.now().strftime("%H:%M")
        with st.spinner("Analysing..."):
            docs    = st.session_state["vector_store"].similarity_search(q, k=top_k)
            sources = [d.page_content for d in docs]
            cv      = conf_score(docs, q)
            answer  = None
            ro      = False

            if st.session_state["qa_chain"]:
                try:
                    res    = st.session_state["qa_chain"].invoke({"query": q})
                    answer = res.get("result", "").strip()
                    if not answer or answer.lower().startswith("not found"):
                        answer = None
                except Exception as e:
                    st.session_state["llm_error"] = str(e)

            if not answer:
                ro     = True
                answer = docs[0].page_content if docs else "No relevant content found."

        st.session_state["chat_history"].append({"role":"user","content":q,"ts":ts})
        st.session_state["chat_history"].append({
            "role":"assistant","content":answer,
            "confidence":cv,"sources":sources,"retrieval_only":ro,"ts":ts,
        })
        st.rerun()

# ── EMPTY STATE ──
else:
    st.markdown(f"""
    <div class="empty">
      <div class="empty-icon">{ICON['doc']}</div>
      <div class="empty-title">Upload a document to begin</div>
      <div class="empty-sub">
        DocMind analyses your PDF, builds a semantic vector index, and lets you
        ask questions in plain English with cited source passages and confidence scores.
      </div>
      <div class="feat-grid">
        <div class="feat-tile">
          <div class="feat-ico">{ICON['summary']}</div>
          <div class="feat-name">Auto Summary</div>
          <div class="feat-desc">Instant overview on upload</div>
        </div>
        <div class="feat-tile">
          <div class="feat-ico">{ICON['brain']}</div>
          <div class="feat-name">LLM Answers</div>
          <div class="feat-desc">Grounded in your document</div>
        </div>
        <div class="feat-tile">
          <div class="feat-ico">{ICON['source']}</div>
          <div class="feat-name">Source Tracing</div>
          <div class="feat-desc">Every answer cited</div>
        </div>
        <div class="feat-tile">
          <div class="feat-ico">{ICON['spark']}</div>
          <div class="feat-name">MMR Retrieval</div>
          <div class="feat-desc">Diverse context chunks</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
