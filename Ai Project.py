import io, csv, string
import pandas as pd
import streamlit as st
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── NLTK Setup ────────────────────────────────
for pkg in ("punkt", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)
STOPS = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

# ─── Text Processing ───────────────────────────
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return " ".join(LEM.lemmatize(w) for w in word_tokenize(text) if w not in STOPS)

def jaccard(a, b):
    a, b = set(word_tokenize(a)), set(word_tokenize(b))
    return len(a & b) / len(a | b) if a | b else 0

def cosine(a, b):
    vecs = CountVectorizer(stop_words="english").fit_transform([a, b]).toarray()
    return cosine_similarity(vecs)[0, 1]

# ─── Load Q&A ──────────────────────────────────
def load_qna(upload):
    df = pd.read_csv(upload, header=None, names=["question", "answer"]).dropna()
    if df.empty: st.error("CSV is empty or malformed."); return []
    return df.to_dict("records")

# ─── Evaluate Answers ──────────────────────────
def evaluate(qna, student):
    feedback, score = [], 0
    for qa, ans in zip(qna, student):
        gold, stu = preprocess(qa["answer"]), preprocess(ans)
        jac, cos = jaccard(gold, stu), cosine(gold, stu)
        sim = (jac + cos) / 2
        if not stu:
            tag, pts = "❌ Incorrect.", 0
        elif sim >= 0.9:
            tag, pts = "✅ Correct!", 1
        elif sim >= 0.5:
            tag, pts = "⚠️ Partially correct.", 0.5
        else:
            tag, pts = "❌ Incorrect.", 0
        score += pts
        feedback.append(f"Q: {qa['question']}\nYour answer: {ans or '[No answer]'}\n{tag}"
                        + ("" if pts else f"\nCorrect answer: {qa['answer']}"))
    return score, feedback

# ─── Streamlit UI ─────────────────────────────
def main():
    st.title("📝 AI Text Assessment Assistant")
    upload = st.file_uploader("Upload Questions & Answers CSV", type="csv")
    if not upload: return
    qna = load_qna(upload); student = []

    for i, q in enumerate(qna, 1):
        st.write(f"Q{i}: {q['question']}")
        student.append(st.text_area(f"Your answer to Q{i}:", height=80))

    if not st.button("Submit"): return

    score, feedback = evaluate(qna, student)
    total = len(qna)
    st.success(f"Score: {score}/{total} ({score / total * 100:.2f}%)")

    st.subheader("Feedback")
    st.write("\n\n".join(feedback))

    # Download feedback
    csv_buf = io.StringIO()
    pd.DataFrame({"Feedback": feedback}).to_csv(csv_buf, index=False)
    st.download_button("Download Feedback", csv_buf.getvalue(), "results.csv")

if __name__ == "__main__":
    main()