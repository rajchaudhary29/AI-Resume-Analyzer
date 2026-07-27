# 📄 AI Resume Analyzer

An AI-powered Resume Analyzer built using **Python, Streamlit, NLP, and Scikit-learn**. This application compares a candidate's resume with a job description using Natural Language Processing (NLP) and TF-IDF vectorization to generate an ATS (Applicant Tracking System) match score along with resume improvement suggestions.

---

## 🚀 Features

- 📂 Upload Resume in PDF format
- 📝 Paste any Job Description
- 📊 ATS Match Score Calculation
- 📈 Resume & Job Description Similarity Analysis
- ✅ Displays Matched Skills
- ❌ Identifies Missing Skills
- 💡 Resume Improvement Suggestions
- 📋 Top Resume Keywords
- 📉 ATS Score Bar Chart
- 🥧 Match Percentage Pie Chart
- 🌙 Modern Dark Theme UI

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- PyPDF2
- Matplotlib

---

## 📂 Project Structure

```
AI-Resume-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_resume.pdf (optional)
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
matplotlib
scikit-learn
PyPDF2
nltk
```

---

## 📖 How It Works

1. Upload your resume in PDF format.
2. Paste the job description.
3. Click **Analyze Match**.
4. The application:
   - Extracts text from the resume.
   - Cleans and preprocesses the text.
   - Removes stop words using NLTK.
   - Converts text into TF-IDF vectors.
   - Calculates cosine similarity.
   - Displays the ATS Match Score.
   - Shows matched and missing skills.
   - Provides resume improvement suggestions.

---

## 📊 Output

- ATS Match Score
- Resume Word Count
- Job Description Word Count
- Matched Skills
- Missing Skills
- Resume Suggestions
- Resume Keywords
- Bar Chart
- Pie Chart

---

## 🎯 Future Improvements

- Support for DOCX resumes
- AI-powered resume rewriting
- Keyword recommendation using LLMs
- Multi-resume comparison
- Job role prediction
- Resume ranking system
- Downloadable analysis report

---

## 👨‍💻 Author

**Raj Chaudhary**

---

## 📜 License
## LINK: https://ai-resume-analyzer-fhcz542hj2d8uh5vyhfewe.streamlit.app/
