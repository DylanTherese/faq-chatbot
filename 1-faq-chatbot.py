import os
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv() 


# Settings and FAQ data
DB_PATH = "chatbot.db"
ADMIN_PASSWORD = "admin123"

# 22M params, English-only. 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

THRESHOLD = 0.6    # customer question vs FAQ -> "I don't know" below int
GROUP_THRESHOLD = 0.5  # unanswered question vs unanswered question -> merge above int
MAX_CHARS = 300 
UNKNOWN_REPLY = (
    "I don't know the answer to that yet. "
    "I've recorded your question so it can be reviewed"
)

FAQS = [
    {
        "question": "What are your opening hours?",
        "answer": "We are open Monday to Friday, 9am to 5pm.",
    },
    {
        "question": "How long does delivery take?",
        "answer": "Orders are usually delivered within 1-3 business days.",
    },
    {
        "question": "Can I pay when my order arrives?",
        "answer": "No.",
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept credit cards, bank transfer, and mobile payments.",
    },
    {
        "question": "How can I pay?",
        "answer": "We accept credit cards, bank transfer, and mobile payments.",
    }
]



# Embedding model
@st.cache_resource(show_spinner="Loading language model (first run downloads it)...")
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def faq_embeddings():
    return get_model().encode([f["question"] for f in FAQS], normalize_embeddings=True)


@st.cache_data(max_entries=50)
def embed_texts(texts: tuple):
    return get_model().encode(list(texts), normalize_embeddings=True)


#  Database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asked_at TEXT,
                question TEXT,
                normalized TEXT,
                answered INTEGER,
                matched_faq TEXT,
                similarity REAL
            )"""
        )

        cols = [r[1] for r in conn.execute("PRAGMA table_info(questions)")]
        if "similarity" not in cols:
            conn.execute("ALTER TABLE questions ADD COLUMN similarity REAL")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def log_question(question: str, answered: bool, matched_faq, similarity: float):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO questions "
            "(asked_at, question, normalized, answered, matched_faq, similarity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                normalize(question),
                int(answered),
                matched_faq,
                round(similarity, 3),
            ),
        )


def load_questions() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM questions", conn)



# Matching and grouping logic
def get_answer(user_input: str):
    """Returns (reply, matched_faq_question or None, similarity)."""
    query = get_model().encode(user_input, normalize_embeddings=True)
    sims = faq_embeddings() @ query  # cosine similarity (vectors are normalized)
    i = int(np.argmax(sims))
    best = float(sims[i])
    if best < THRESHOLD:
        return UNKNOWN_REPLY, None, best
    return FAQS[i]["answer"], FAQS[i]["question"], best


def group_similar(unanswered: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Merge differently-worded versions of the same unanswered question."""
    counts = (
        unanswered.groupby("normalized")
        .agg(question=("question", "first"), n=("id", "count"), last=("asked_at", "max"))
        .sort_values("n", ascending=False)
    )
    vecs = embed_texts(tuple(counts.index))

    groups, centers = [], []  # a group's center is its first (most asked) question
    for i, (_, row) in enumerate(counts.iterrows()):
        sims = [float(vecs[i] @ c) for c in centers]
        if sims and max(sims) >= threshold:
            g = groups[int(np.argmax(sims))]
            g["times_asked"] += int(row["n"])
            g["variants"].append(row["question"])
            g["last_asked"] = max(g["last_asked"], row["last"])
        else:
            centers.append(vecs[i])
            groups.append(
                {
                    "question": row["question"],
                    "times_asked": int(row["n"]),
                    "variants": [row["question"]],
                    "last_asked": row["last"],
                }
            )

    out = pd.DataFrame(groups)
    out["variants"] = out["variants"].apply(lambda v: " | ".join(v))
    return out.sort_values("times_asked", ascending=False).reset_index(drop=True)


# Customer page
def customer_page():
    st.title("💬 FAQ Chatbot")
    st.caption("Ask me a question and I'll look for the best answer.")

    with st.sidebar:
        st.header("Common questions")
        for f in FAQS:
            st.write("•", f["question"])

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type your question...")
    if not prompt or not prompt.strip():
        return
    prompt = prompt.strip()
    
    if len(prompt) > MAX_CHARS:
        st.warning(f"Please keep your question under {MAX_CHARS} characters.")
        return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        reply, matched, sim = get_answer(prompt)
    log_question(prompt, answered=matched is not None, matched_faq=matched, similarity=sim)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

# Admin page
def admin_page():
    st.title("📊 Admin dashboard")

    if not st.session_state.get("is_admin"):
        pwd = st.text_input("Admin password: admin123", type="password")
        if st.button("Log in"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Wrong password. Password: admin123")
        return

    if st.sidebar.button("Log out"):
        st.session_state.is_admin = False
        st.rerun()

    df = load_questions()
    if df.empty:
        st.info("No questions yet. Ask something on the Customer page first.")
        return

    total = len(df)
    unanswered = df[df["answered"] == 0]
    answered_count = total - len(unanswered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total questions", total)
    c2.metric("Answered", answered_count)
    c3.metric("Unanswered", len(unanswered))
    c4.metric("Answer rate", f"{answered_count / total:.0%}")

    st.subheader("Unanswered questions (most asked first)")
    if unanswered.empty:
        st.success("The bot answered everything so far.")
    else:
        summary = group_similar(unanswered, GROUP_THRESHOLD)
        st.caption(
            f"{len(unanswered)} unanswered messages grouped into "
            f"{len(summary)} distinct questions."
        )
        st.dataframe(summary, use_container_width=True)
        st.bar_chart(summary.head(10).set_index("question")["times_asked"])

    st.subheader("Most asked FAQ questions")
    matched = df[df["answered"] == 1]
    if matched.empty:
        st.write("Nothing yet.")
    else:
        st.bar_chart(matched["matched_faq"].value_counts())

    with st.expander("Full question log"):
        st.dataframe(df.sort_values("id", ascending=False), use_container_width=True)
        st.download_button(
            "Download log as CSV",
            df.to_csv(index=False),
            file_name="question_log.csv",
            mime="text/csv",
        )


# App entry point
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
init_db()

page = st.sidebar.radio("View", ["Customer", "Admin"])
if page == "Customer":
    customer_page()
else:
    admin_page()