import streamlit as st
import matplotlib.pyplot as plt
from sklearn .feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")

# Page Setup

st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")
st.markdown("""
<style>


/* Main background */
.stApp{
    background-color:#0E1117;
    color:white;
}

/* All headings */
h1,h2,h3,h4,h5,h6{
    color:white !important;
}

/* Paragraphs & labels */
p,label,span{
    color:white !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#161B22;
}

[data-testid="stSidebar"] *{
    color:white !important;
}

/* File uploader */
[data-testid="stFileUploader"]{
    background:#1F2937;
    border:2px dashed #4CAF50;
    border-radius:12px;
    padding:15px;
}

[data-testid="stFileUploader"] *{
    color:white !important;
}

/* Text area */
textarea{
    background:#1F2937 !important;
    color:white !important;
    border:1px solid #4CAF50 !important;
}

/* Text input */
input{
    background:#1F2937 !important;
    color:white !important;
}

/* Buttons */
.stButton>button{
    background:#00C853;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#00E676;
    color:black;
}

/* Metrics */
[data-testid="stMetric"]{
    background:#1F2937;
    padding:15px;
    border-radius:12px;
}

/* Success */
[data-testid="stAlert"]{
    border-radius:10px;
}

/* Progress bar */
.stProgress>div>div>div{
    background:#00C853;
}

</style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer")

st.markdown("""
### Smart Resume Screening using NLP


""")

with st.sidebar:

    st.title("📄 AI Resume Analyzer")

    st.markdown("---")

    st.markdown("""

""")

    st.markdown("---")

    st.info("""
Built with

- Python
- Streamlit
- Scikit-Learn
- NLTK
- PyPDF2
- Matplotlib
""")
    # helper function


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reder = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reder.pages:
            text = text + page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF:{e}")
        return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed


# def extract_keywords(text,num_keywords=10):
#     words=word_tokenize(text)
#     words=[w for w in words if len(w)>2]
#     tagged_words=pos_tag(words)
#     nouns=[w for w,pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
#     word_freq=Counter(nouns)
#     return word_freq.most_common(num_keywords)


# Main aap

def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    job_description = st.text_area("Paste the job description", height=200)

    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste the job description")
            return

        with st.spinner("Analyzing your resume...."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("could not extract text from pdf. please try another pdf")
                return

                # calculate similarity
            similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
            resume_words = resume_processed.split()
            job_words = job_processed.split()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Resume Words", len(resume_words))

            with col2:
                st.metric("Job Words", len(job_words))

            with col3:
                st.metric("ATS Score", f"{similarity_score:.1f}%")

            # Result
            st.subheader("ATS Screening Result")

            st.metric("Match Score", f"{similarity_score:.2f}%")

            st.progress(similarity_score / 100)


            if similarity_score >= 85:
                st.success("🟢 Excellent ATS Match")

            elif similarity_score >= 70:
                st.info("🟡 Good ATS Match")

            elif similarity_score >= 50:
                st.warning("🟠 Average ATS Match")

            else:
                st.error("🔴 Poor ATS Match")

            # gauge chart /bar chart

            fig, ax = plt.subplots(figsize=(8, 1))

            ax.barh(
                ["ATS Score"],
                [similarity_score],
                color="#00C853"
            )

            ax.set_xlim(0, 100)

            for i in [25, 50, 75]:
                ax.axvline(i, color="gray", linestyle="--")

            st.pyplot(fig)
             #pie chart
            fig2, ax2 = plt.subplots(figsize=(5, 5))

            ax2.pie(
                [similarity_score, 100 - similarity_score],
                labels=["Matched", "Missing"],
                autopct="%1.1f%%",
                startangle=90
            )

            st.pyplot(fig2)
             #resume matched skills
            resume_set = set(resume_processed.split())
            job_set = set(job_processed.split())

            matched = resume_set.intersection(job_set)

            st.subheader("✅ Matched Skills")

            if matched:
                st.write(", ".join(sorted(list(matched))[:30]))
            else:
                st.write("No matching skills found.")
       #missing skills
            missing = job_set - resume_set

            st.subheader("❌ Missing Skills")

            if missing:
                st.write(", ".join(sorted(list(missing))[:30]))
            else:
                st.success("No important skills are missing.")
            #resume suggestion
            st.subheader("💡 Resume Suggestions")

            if similarity_score < 50:

                st.write("""
            - Add more technical skills.
            - Include projects related to the job.
            - Improve your experience section.
            - Use keywords from the job description.
            """)

            elif similarity_score < 80:

                st.write("""
            - Add more job-specific keywords.
            - Highlight achievements using numbers.
            - Expand project descriptions.
            """)

            else:

                st.success("Excellent resume. Only minor improvements needed.")
            #resume suggestion
            st.subheader("📊 Top Resume Keywords")

            counter = Counter(resume_processed.split())

            for word, count in counter.most_common(10):
                st.write(f"**{word}** : {count}")
            if similarity_score < 40:
                st.warning("Low Match, consider tailoring your resume more closely.")
            elif similarity_score < 70:
                st.info("Good Match. Your resume aligin fairly well")
            else:
                st.success("Excellent Match ! Your resume strongly aligns.")


if __name__ == "__main__":
    main()