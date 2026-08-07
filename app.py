import re
from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# --------------------------------------------------------------------------
# NLTK setup (cached so it only runs once per session, not on every rerun)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def setup_nltk():
    for pkg, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return set(stopwords.words("english")), WordNetLemmatizer()


STOPWORDS, LEMMATIZER = setup_nltk()

# --------------------------------------------------------------------------
# Curated skills taxonomy. Matching against a real vocabulary (instead of
# raw "every word that isn't in the other document") is what actually makes
# matched/missing skills meaningful rather than noisy leftover words.
# --------------------------------------------------------------------------

SKILLS_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala",
        "r", "matlab", "sql", "bash", "shell scripting",
    ],
    "Web & Frameworks": [
        "react", "react.js", "angular", "vue", "vue.js", "node.js", "nodejs",
        "express.js", "django", "flask", "fastapi", "spring", "spring boot",
        "next.js", ".net", "asp.net", "html", "css", "tailwind", "bootstrap",
        "rest api", "graphql", "streamlit",
    ],
    "Data & ML": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "pandas", "numpy", "scikit-learn", "sklearn",
        "tensorflow", "pytorch", "keras", "data analysis", "data visualization",
        "matplotlib", "seaborn", "power bi", "tableau", "statistics",
        "data science", "llm", "large language models", "generative ai",
    ],
    "Databases": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "oracle",
        "redis", "elasticsearch", "dynamodb", "cassandra", "nosql",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "ci/cd", "jenkins", "terraform", "ansible", "git", "github", "gitlab",
        "linux", "devops", "microservices",
    ],
    "Tools & Platforms": [
        "jira", "confluence", "slack", "excel", "figma", "postman",
        "airflow", "spark", "hadoop", "kafka",
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "agile", "scrum",
        "time management", "collaboration", "stakeholder management",
        "mentoring", "presentation",
    ],
}

ALL_SKILLS = [s for group in SKILLS_TAXONOMY.values() for s in group]
SKILL_TO_CATEGORY = {
    s: cat for cat, skills in SKILLS_TAXONOMY.items() for s in skills
}

# Sort longest-first so multi-word skills (e.g. "machine learning") are
# matched before their substrings would otherwise get missed.
_SKILL_PATTERN = re.compile(
    r"(?<![\w.+#])("
    + "|".join(sorted((re.escape(s) for s in ALL_SKILLS), key=len, reverse=True))
    + r")(?![\w.+#])",
    re.IGNORECASE,
)


def extract_skills(raw_text: str) -> set:
    """Match against the curated taxonomy on the ORIGINAL text, before any
    punctuation stripping, so tokens like 'C++', 'C#', 'Node.js' survive."""
    found = {m.group(1).lower() for m in _SKILL_PATTERN.finditer(raw_text)}
    return found


# --------------------------------------------------------------------------
# Text processing
# --------------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    from nltk.tokenize import word_tokenize

    words = word_tokenize(clean_text(text))
    lemmas = [
        LEMMATIZER.lemmatize(w) for w in words
        if w not in STOPWORDS and len(w) > 2
    ]
    return " ".join(lemmas)


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


@st.cache_data(show_spinner=False)
def text_similarity(resume_processed: str, job_processed: str) -> float:
    # Bigrams + sublinear tf give a noticeably better signal than raw
    # unigram counts on short documents like resumes/JDs.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf = vectorizer.fit_transform([resume_processed, job_processed])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)


def top_job_terms(job_processed: str, resume_processed: str, n=12):
    """Job-description terms with the highest TF-IDF weight that are absent
    from the resume — a better 'what to add' list than plain set difference."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf = vectorizer.fit_transform([job_processed, resume_processed])
    feature_names = vectorizer.get_feature_names_out()
    job_scores = tfidf[0].toarray()[0]
    resume_terms = set(resume_processed.split())
    ranked = sorted(zip(feature_names, job_scores), key=lambda x: -x[1])
    missing = [
        term for term, score in ranked
        if score > 0 and term not in resume_terms and term not in ALL_SKILLS
    ]
    return missing[:n]


# --------------------------------------------------------------------------
# Page setup / styling
# --------------------------------------------------------------------------

st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
.stApp{background-color:#0E1117;color:white;}
h1,h2,h3,h4,h5,h6{color:white !important;}
p,label,span{color:white !important;}
[data-testid="stSidebar"]{background-color:#161B22;}
[data-testid="stSidebar"] *{color:white !important;}
[data-testid="stFileUploader"]{background:#1F2937;border:2px dashed #4CAF50;border-radius:12px;padding:15px;}
[data-testid="stFileUploader"] *{color:white !important;}
textarea{background:#1F2937 !important;color:white !important;border:1px solid #4CAF50 !important;}
input{background:#1F2937 !important;color:white !important;}
.stButton>button{background:#00C853;color:white;border:none;border-radius:10px;font-weight:bold;}
.stButton>button:hover{background:#00E676;color:black;}
[data-testid="stMetric"]{background:#1F2937;padding:15px;border-radius:12px;}
[data-testid="stAlert"]{border-radius:10px;}
.stProgress>div>div>div{background:#00C853;}
.skill-chip{display:inline-block;background:#1F2937;border:1px solid #00C853;border-radius:20px;
    padding:4px 12px;margin:3px;font-size:0.85rem;}
.skill-chip-missing{border-color:#FF5252;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📄 AI Resume Analyzer")
st.markdown("### Smart Resume Screening using NLP")

with st.sidebar:
    st.title("📄 AI Resume Analyzer")
    st.markdown("---")
    with st.expander("ℹ️ How the score works"):
        st.write(
            "**Skill Match** — how many of the job's required skills, tools "
            "and technologies (from a curated taxonomy) also appear in your "
            "resume.\n\n"
            "**Text Similarity** — TF-IDF cosine similarity between the two "
            "documents, capturing overall wording/context overlap.\n\n"
            "**ATS Score** is a weighted blend: real ATS systems key mostly "
            "on exact keyword/skill matches, so skill match counts for 70% "
            "and text similarity for 30%."
        )
    st.markdown("---")
    st.info(
        "Built with\n\n- Python\n- Streamlit\n- Scikit-Learn\n- NLTK\n- pypdf\n- Matplotlib"
    )

# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------


def render_score_gauge(score):
    fig, ax = plt.subplots(figsize=(8, 1.1))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    color = "#00C853" if score >= 70 else "#FFC107" if score >= 50 else "#FF5252"
    ax.barh(["ATS Score"], [score], color=color, height=0.5)
    ax.set_xlim(0, 100)
    for i in [25, 50, 75]:
        ax.axvline(i, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="white")
    st.pyplot(fig, use_container_width=True)


def render_skill_donut(matched_count, missing_count):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_alpha(0)
    if matched_count + missing_count == 0:
        ax.text(0.5, 0.5, "No skills detected", ha="center", va="center", color="white")
        ax.axis("off")
    else:
        ax.pie(
            [matched_count, missing_count],
            labels=["Matched", "Missing"],
            colors=["#00C853", "#FF5252"],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"width": 0.4},
            textprops={"color": "white"},
        )
    st.pyplot(fig, use_container_width=True)


def build_report(similarity, skill_pct, ats_score, matched, missing, suggestions):
    lines = [
        "RESUME / JOB MATCH REPORT",
        "=" * 30,
        f"ATS Score: {ats_score:.1f}%",
        f"Skill Match: {skill_pct:.1f}%",
        f"Text Similarity: {similarity:.1f}%",
        "",
        f"Matched Skills ({len(matched)}): {', '.join(sorted(matched)) or 'None'}",
        "",
        f"Missing Skills ({len(missing)}): {', '.join(sorted(missing)) or 'None'}",
        "",
        "Suggestions:",
    ] + [f"- {s}" for s in suggestions]
    return "\n".join(lines)


def main():
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    with col_b:
        job_description = st.text_area("Paste the job description", height=200)

    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description.strip():
            st.warning("Please paste the job description")
            return

        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file.getvalue())
            if not resume_text.strip():
                st.error("Could not extract text from PDF. Please try another file.")
                return

            resume_processed = preprocess(resume_text)
            job_processed = preprocess(job_description)

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_description)
            matched = resume_skills & job_skills
            missing = job_skills - resume_skills

            similarity = text_similarity(resume_processed, job_processed)
            skill_pct = round(100 * len(matched) / len(job_skills), 2) if job_skills else 100.0
            ats_score = round(0.7 * skill_pct + 0.3 * similarity, 2)

        # ---- Top metrics ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATS Score", f"{ats_score:.1f}%")
        c2.metric("Skill Match", f"{skill_pct:.1f}%")
        c3.metric("Text Similarity", f"{similarity:.1f}%")
        c4.metric("Skills Found in JD", len(job_skills))

        st.progress(min(ats_score, 100) / 100)

        if ats_score >= 80:
            st.success("🟢 Excellent ATS Match")
        elif ats_score >= 65:
            st.info("🟡 Good ATS Match")
        elif ats_score >= 45:
            st.warning("🟠 Average ATS Match")
        else:
            st.error("🔴 Poor ATS Match")

        tab_overview, tab_skills, tab_keywords, tab_tips = st.tabs(
            ["📊 Overview", "🧩 Skills", "🔑 Keywords", "💡 Suggestions"]
        )

        with tab_overview:
            g1, g2 = st.columns([2, 1])
            with g1:
                st.subheader("Score Breakdown")
                render_score_gauge(ats_score)
            with g2:
                st.subheader("Skill Coverage")
                render_skill_donut(len(matched), len(missing))

        with tab_skills:
            st.subheader(f"✅ Matched Skills ({len(matched)})")
            if matched:
                chips = "".join(
                    f'<span class="skill-chip">{s}</span>' for s in sorted(matched)
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.write("No matching skills detected from the taxonomy.")

            st.markdown("---")
            st.subheader(f"❌ Missing Skills ({len(missing)})")
            if missing:
                chips = "".join(
                    f'<span class="skill-chip skill-chip-missing">{s}</span>'
                    for s in sorted(missing)
                )
                st.markdown(chips, unsafe_allow_html=True)

                by_category = {}
                for s in missing:
                    by_category.setdefault(SKILL_TO_CATEGORY.get(s, "Other"), []).append(s)
                st.markdown("---")
                st.caption("Missing skills by category")
                for cat, items in sorted(by_category.items()):
                    st.write(f"**{cat}**: {', '.join(sorted(items))}")
            else:
                st.success("No important skills are missing.")

        with tab_keywords:
            st.subheader("📊 Top Resume Keywords")
            counter = Counter(resume_processed.split())
            for word, count in counter.most_common(10):
                st.write(f"**{word}** : {count}")

            st.markdown("---")
            st.subheader("🔎 High-value JD terms not in your resume")
            extra_terms = top_job_terms(job_processed, resume_processed)
            if extra_terms:
                st.write(", ".join(extra_terms))
            else:
                st.write("Your resume already covers the JD's key terms well.")

        with tab_tips:
            st.subheader("💡 Resume Suggestions")
            suggestions = []
            if missing:
                top_missing = sorted(missing)[:8]
                suggestions.append(
                    f"Work these missing skills into your resume where genuinely applicable: {', '.join(top_missing)}."
                )
            if skill_pct < 50:
                suggestions += [
                    "Add more of the technical skills and tools the job actually lists.",
                    "Include projects or experience that map directly to the role's requirements.",
                ]
            elif skill_pct < 80:
                suggestions.append("Weave a few more job-specific keywords naturally into your bullet points.")
            if similarity < 50:
                suggestions.append("Mirror more of the job description's language/terminology in your summary and experience sections.")
            suggestions.append("Quantify achievements with numbers (%, $, time saved) wherever possible.")
            if ats_score >= 80:
                suggestions = ["Excellent match — only minor tailoring needed."] + suggestions

            for s in suggestions:
                st.write(f"- {s}")

            st.markdown("---")
            report = build_report(similarity, skill_pct, ats_score, matched, missing, suggestions)
            st.download_button(
                "⬇️ Download Report",
                data=report,
                file_name="resume_match_report.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()