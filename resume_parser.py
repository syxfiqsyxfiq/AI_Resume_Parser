import pdfplumber
import docx
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

def parse_resume(text):
    doc = nlp(text)

    # Extract name (first PERSON entity)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Extract email
    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    email = email.group(0) if email else None

    # Extract phone number (basic pattern)
    phone = re.search(r"(\+?\d{1,3}[-.\s]?)?\d{10}", text)
    phone = phone.group(0) if phone else None

    # Extract skills (simple dictionary lookup)
    skills_db = ["python", "java", "sql", "machine learning", "nlp", "excel", "communication"]
    skills = [skill for skill in skills_db if skill.lower() in text.lower()]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills
    }

def match_resume(resume_texts, job_description):
    docs = resume_texts + [job_description]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    return similarity.flatten()  # scores for each resume

if __name__ == "__main__":
    # Replace with your own resumes
    resume1 = extract_text("resume1.pdf")
    resume2 = extract_text("resume2.docx")

    parsed1 = parse_resume(resume1)
    parsed2 = parse_resume(resume2)

    print("Resume 1 Parsed Data:", parsed1)
    print("Resume 2 Parsed Data:", parsed2)

    # Job description example
    job_desc = "Looking for Python developer with SQL experience"
    scores = match_resume([resume1, resume2], job_desc)

    print("Relevance Scores:", scores)
