import streamlit as st
import pdfplumber
import docx
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# Resume Text Extraction
# ----------------------------
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# ----------------------------
# Resume Parsing
# ----------------------------
def parse_resume(text):
    doc = nlp(text)

    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    email = email.group(0) if email else None

    phone = re.search(r"(\+?\d{1,3}[-.\s]?)?\d{7,12}", text)
    phone = phone.group(0) if phone else None

    skills_db = ["python", "java", "sql", "machine learning", "nlp", "excel", "communication"]
    skills = [skill for skill in skills_db if skill.lower() in text.lower()]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills
    }

# ----------------------------
# Matching Function
# ----------------------------
def match_resume(resume_texts, job_description):
    docs = resume_texts + [job_description]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    return similarity.flatten()

# ----------------------------
# Generate PDF Report
# ----------------------------
def create_pdf_report(report_text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 12)
    width, height = A4

    y = height - inch
    for line in report_text.split("\n"):
        if y < inch:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = height - inch
        pdf.drawString(1 * inch, y, line)
        y -= 15

    pdf.save()
    buffer.seek(0)
    return buffer

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="AI Resume Parser Chat", layout="wide")
st.title("💬 AI Resume Parser Chat")

# --- Sidebar ---
st.sidebar.title("🧭 Controls")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = None
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# --- Clear Chat Button in Sidebar ---
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.confirm_clear = True

if st.session_state.confirm_clear:
    st.sidebar.warning("⚠️ Are you sure you want to clear all chat and files?")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Yes, clear all"):
            st.session_state.messages = []
            st.session_state.files = None
            st.session_state.confirm_clear = False
            st.experimental_rerun()
    with col2:
        if st.button("❌ Cancel"):
            st.session_state.confirm_clear = False

# --- Chat History in Sidebar ---
if st.sidebar.checkbox("📜 Show Chat History", value=False):
    st.sidebar.write("### Chat Messages:")
    for msg in st.session_state.messages:
        st.sidebar.markdown(f"**{msg['role'].capitalize()}:** {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")

# --- Download Report in Sidebar ---
if st.sidebar.button("📥 Download Report"):
    last_reply = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            last_reply = msg["content"]
            break

    if last_reply:
        pdf_file = create_pdf_report(last_reply)
        st.sidebar.download_button(
            label="⬇️ Click to Download PDF Report",
            data=pdf_file,
            file_name="resume_analysis_report.pdf",
            mime="application/pdf"
        )
    else:
        st.sidebar.warning("⚠️ No analysis found yet. Run a job description first.")

# --- File Upload (main area) ---
uploaded_files = st.file_uploader("📎 Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    st.session_state.files = uploaded_files
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

# --- Chat Section ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type a job description or request..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.files:
        resume_texts, parsed_data = [], []
        for file in st.session_state.files:
            text = extract_text(file)
            resume_texts.append(text)
            parsed_data.append(parse_resume(text))

        scores = match_resume(resume_texts, user_input)
        for i, parsed in enumerate(parsed_data):
            parsed["score"] = scores[i]
        ranked = sorted(parsed_data, key=lambda x: x["score"], reverse=True)

        reply = "📊 **Candidate Ranking (Best → Worst):**\n\n"
        for rank, parsed in enumerate(ranked, start=1):
            reply += f"**{rank}. {parsed['name'] or 'Unknown'}**\n"
            reply += f"- 📧 {parsed['email'] or 'N/A'}\n"
            reply += f"- 📱 {parsed['phone'] or 'N/A'}\n"
            reply += f"- 🛠 Skills: {', '.join(parsed['skills']) if parsed['skills'] else 'N/A'}\n"
            reply += f"- ✅ Match Score: {parsed['score']:.2f}\n\n"
    else:
        reply = "⚠️ Please upload at least one resume before analysis."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
