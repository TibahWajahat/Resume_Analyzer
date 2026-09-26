"""
AI Resume Analyzer & Job Matcher
Created by Tibah Wajahat | CSE - Data Science

A rule-based (no external AI API) Streamlit app that analyzes a resume
PDF against a pasted job description and produces an ATS-style score,
job-match percentage, skill gap analysis, best-fit role estimation,
keyword analysis, resume section detection, strengths, interview
question prep, a generated professional summary, and a downloadable
report.
"""

import re
import io
from collections import Counter
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PyPDF2 import PdfReader


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown(
    """
    <style>
        .main > div { padding-top: 1.5rem; }

        .app-header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
        }
        .app-header h1 {
            color: #ffffff;
            font-size: 2.1rem;
            margin-bottom: 0.3rem;
        }
        .app-header p {
            color: #ede9fe;
            font-size: 1.02rem;
            margin: 0;
        }

        .metric-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            border: 1px solid #eef0f4;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
            text-align: center;
        }
        .metric-card .value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #4f46e5;
        }
        .metric-card .label {
            font-size: 0.85rem;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .section-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            border: 1px solid #eef0f4;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }

        .pill {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.18rem;
        }
        .pill-matched { background: #d1fae5; color: #065f46; }
        .pill-missing { background: #fee2e2; color: #991b1b; }
        .pill-neutral { background: #e0e7ff; color: #3730a3; }

        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.85rem;
            padding: 1.5rem 0 0.5rem 0;
            border-top: 1px solid #eef0f4;
            margin-top: 2rem;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            font-weight: 700;
            border-radius: 10px;
            border: none;
            padding: 0.7rem 1rem;
            font-size: 1.05rem;
        }
        div.stButton > button:hover {
            box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# STATIC DATA: SKILLS, ROLES, SECTIONS, INTERVIEW QUESTIONS, STOPWORDS
# =============================================================================
SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "c", "c++", "c#", "javascript", "r", "sql",
    ],
    "Web Development": [
        "html", "css", "react", "node.js", "flask", "django", "rest api",
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "oracle",
    ],
    "Data Science & ML": [
        "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
        "tensorflow", "pytorch", "machine learning", "deep learning",
        "data science", "data analysis", "data visualization", "nlp",
        "natural language processing", "statistics",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "linux",
    ],
    "BI & Tools": [
        "excel", "power bi", "tableau", "streamlit", "jupyter",
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "time management", "project management", "agile",
    ],
}
SKILLS = sorted({s for group in SKILL_CATEGORIES.values() for s in group})

ROLE_SKILLS = {
    "Data Analyst": {"sql", "excel", "power bi", "tableau", "python", "pandas",
                      "data analysis", "data visualization", "statistics"},
    "Data Scientist": {"python", "machine learning", "deep learning", "data science",
                        "pandas", "numpy", "scikit-learn", "sql", "statistics",
                        "data visualization"},
    "Python Developer": {"python", "flask", "django", "rest api", "git", "sql", "github"},
    "ML Engineer": {"python", "machine learning", "deep learning", "tensorflow",
                     "pytorch", "scikit-learn", "docker", "aws", "git"},
    "Web Developer": {"html", "css", "javascript", "react", "node.js", "flask",
                       "django", "git", "rest api"},
    "Software Developer": {"java", "c++", "python", "git", "github", "sql",
                            "rest api", "agile"},
    "Business Analyst": {"excel", "power bi", "tableau", "sql", "communication",
                          "data analysis", "project management", "agile"},
}

SECTION_PATTERNS = {
    "Contact": r"(contact\s*information|contact\s*details|email|phone|linkedin)",
    "Summary": r"(summary|objective|profile|about\s*me)",
    "Education": r"(education|academic\s*background|qualification)",
    "Skills": r"(technical\s*skills|skills|core\s*competenc)",
    "Projects": r"(projects|personal\s*projects|academic\s*projects)",
    "Experience": r"(experience|internship|work\s*history|employment)",
    "Certifications": r"(certification|certificate|licenses)",
    "Achievements": r"(achievements|awards|accomplishments|honors)",
}

INTERVIEW_QUESTIONS = {
    "python": ["What is the difference between a list and a tuple in Python?",
               "Explain how Python's garbage collection works."],
    "sql": ["What is the difference between INNER JOIN and LEFT JOIN?",
            "How would you optimize a slow-running SQL query?"],
    "java": ["Explain the difference between an interface and an abstract class in Java.",
             "What is the purpose of the 'final' keyword?"],
    "c++": ["What is the difference between a pointer and a reference in C++?",
            "Explain the concept of virtual functions."],
    "javascript": ["Explain the difference between '==' and '===' in JavaScript.",
                   "What is a closure in JavaScript?"],
    "html": ["What is semantic HTML and why does it matter?"],
    "css": ["Explain the CSS box model."],
    "machine learning": ["What is the bias-variance tradeoff?",
                          "How do you handle overfitting in a model?"],
    "deep learning": ["What is the vanishing gradient problem and how is it addressed?"],
    "data science": ["Walk me through your typical data analysis workflow.",
                      "How do you handle missing or inconsistent data?"],
    "tensorflow": ["What is the difference between a TensorFlow tensor and a NumPy array?"],
    "pytorch": ["How does autograd work in PyTorch?"],
    "nlp": ["What is the difference between stemming and lemmatization?"],
    "aws": ["Which AWS services have you used, and for what purpose?"],
    "azure": ["What Azure services are you familiar with?"],
    "docker": ["What is the difference between a Docker image and a container?"],
    "git": ["What is the difference between 'git merge' and 'git rebase'?"],
    "excel": ["What is the difference between VLOOKUP and INDEX-MATCH?"],
    "power bi": ["What is the difference between a calculated column and a measure in Power BI?"],
    "tableau": ["What types of joins can you perform in Tableau?"],
    "communication": ["Describe a time you had to explain a technical concept to a non-technical audience."],
    "leadership": ["Tell me about a time you led a team through a challenging project."],
    "teamwork": ["Describe a situation where you disagreed with a teammate. How did you resolve it?"],
    "problem solving": ["Walk me through how you approach debugging a difficult issue."],
}
HR_QUESTIONS = [
    "Tell me about yourself and why you're interested in this role.",
    "What are your greatest strengths and weaknesses?",
    "Why should we hire you over other candidates?",
    "Where do you see yourself in the next few years?",
    "Describe a challenge you faced and how you overcame it.",
]

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves will using use used work works looking
strong good great excellent proven ability role team years experience
skill skills plus required requirement requirements preferred nice
""".split())


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file object. Returns '' on failure."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception:
        return ""


def find_skills(text: str, skill_list=None) -> list:
    """Return sorted list of skills (from skill_list) present in text."""
    if skill_list is None:
        skill_list = SKILLS
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def detect_sections(text: str) -> dict:
    """Return dict of section_name -> bool, whether detected in resume text."""
    text_lower = text.lower()
    result = {}
    for section, pattern in SECTION_PATTERNS.items():
        result[section] = bool(re.search(pattern, text_lower))
    return result


def calculate_job_match(resume_skills: list, job_skills: list) -> float:
    if not job_skills:
        return 0.0
    matched = set(resume_skills).intersection(set(job_skills))
    return round((len(matched) / len(set(job_skills))) * 100, 1)


def best_fit_roles(resume_skills: list) -> list:
    """Return list of (role, percent) sorted descending by fit percentage."""
    resume_set = set(resume_skills)
    scores = []
    for role, req_skills in ROLE_SKILLS.items():
        if not req_skills:
            continue
        pct = (len(resume_set.intersection(req_skills)) / len(req_skills)) * 100
        scores.append((role, round(pct, 1)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def extract_keywords(job_text: str, exclude: set, top_n: int = 15) -> list:
    """Extract frequent, meaningful words from the job description.

    `exclude` should contain full skill phrases (e.g. "power bi"); this also
    excludes each individual word within those phrases (e.g. "power", "bi")
    so keywords don't just repeat words already covered by skill matching.
    """
    exclude_words = set()
    for phrase in exclude:
        exclude_words.update(re.findall(r"[a-zA-Z]+", phrase.lower()))

    words = re.findall(r"[a-zA-Z]{4,}", job_text.lower())
    filtered = [w for w in words if w not in STOPWORDS and w not in exclude_words]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(top_n)]


def keyword_analysis(resume_text: str, job_description: str, skills_already_used: set) -> tuple:
    keywords = extract_keywords(job_description, skills_already_used)
    resume_lower = resume_text.lower()
    found, missing = [], []
    for kw in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
        if re.search(pattern, resume_lower):
            found.append(kw)
        else:
            missing.append(kw)
    return found, missing


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def compute_ats_score(resume_text: str, sections: dict, resume_skills: list,
                       job_skills: list, keyword_found: list, keyword_total: int) -> dict:
    """Return a dict with total score (0-100) and its component breakdown."""
    # 1) Skill match with job description - up to 40 points
    job_match_pct = calculate_job_match(resume_skills, job_skills) if job_skills else 0
    skills_component = round((job_match_pct / 100) * 40, 1) if job_skills else round(
        min(len(resume_skills) / 15, 1.0) * 40, 1
    )

    # 2) Resume completeness (sections present) - up to 25 points
    sections_present = sum(1 for v in sections.values() if v)
    completeness_component = round((sections_present / len(sections)) * 25, 1)

    # 3) Keyword presence - up to 20 points
    if keyword_total > 0:
        keyword_component = round((len(keyword_found) / keyword_total) * 20, 1)
    else:
        keyword_component = 20.0  # nothing specific to check against

    # 4) Content richness / length - up to 15 points
    wc = word_count(resume_text)
    if wc == 0:
        length_component = 0.0
    elif wc < 150:
        length_component = round((wc / 150) * 8, 1)
    elif 150 <= wc <= 900:
        length_component = 15.0
    else:
        # very long resumes lose a little for verbosity
        overflow = min((wc - 900) / 900, 1.0)
        length_component = round(15 - (overflow * 4), 1)

    total = round(skills_component + completeness_component + keyword_component + length_component, 1)
    total = max(0.0, min(100.0, total))

    return {
        "total": total,
        "skills_component": skills_component,
        "completeness_component": completeness_component,
        "keyword_component": keyword_component,
        "length_component": length_component,
        "sections_present": sections_present,
        "word_count": wc,
    }


def generate_strengths(resume_skills: list, sections: dict, ats: dict, job_match_pct: float) -> list:
    strengths = []
    if sections.get("Skills"):
        strengths.append("A dedicated Skills section clearly lists technical capabilities.")
    if sections.get("Projects"):
        strengths.append("Includes a Projects section, showing hands-on, applied experience.")
    if sections.get("Experience"):
        strengths.append("Work/internship experience is documented.")
    if sections.get("Education"):
        strengths.append("Educational background is clearly stated.")
    if sections.get("Certifications"):
        strengths.append("Certifications add credibility to technical claims.")
    if len(resume_skills) >= 10:
        strengths.append(f"Broad technical skill set detected ({len(resume_skills)} skills recognized).")
    for category, skills_in_cat in SKILL_CATEGORIES.items():
        overlap = set(resume_skills).intersection(set(skills_in_cat))
        if len(overlap) >= 3:
            strengths.append(f"Strong footing in {category} ({len(overlap)} related skills detected).")
    if job_match_pct >= 60:
        strengths.append(f"High alignment with the target job description ({job_match_pct}% skill match).")
    if not strengths:
        strengths.append("Resume text was extracted, but no strong signals were detected yet — "
                          "consider adding more detail to Skills and Projects sections.")
    return strengths


def generate_suggestions(missing_skills: list, sections: dict, ats: dict, keyword_missing: list) -> list:
    suggestions = []
    if missing_skills:
        shown = ", ".join(s.title() for s in missing_skills[:8])
        suggestions.append(
            f"If you genuinely have experience with these, consider adding them: {shown}."
        )
    for section, present in sections.items():
        if not present and section in ("Skills", "Projects", "Experience", "Education"):
            suggestions.append(f"No '{section}' section was detected — consider adding one if applicable.")
    if ats["length_component"] < 10:
        if ats["word_count"] < 150:
            suggestions.append("Your resume content seems short — consider adding more detail on projects, "
                                "responsibilities, and outcomes.")
        else:
            suggestions.append("Your resume is quite long — consider trimming it for conciseness and impact.")
    if keyword_missing:
        shown = ", ".join(keyword_missing[:6])
        suggestions.append(f"These job-description keywords were not detected in your resume: {shown}.")
    if not suggestions:
        suggestions.append("Your resume already covers the key elements detected in the job description well.")
    return suggestions


def generate_summary(resume_skills: list, sections: dict, top_role: str) -> str:
    if not resume_skills and not any(sections.values()):
        return ("Not enough information was reliably detected in the uploaded resume to generate "
                "a summary. Please ensure the PDF contains selectable (non-scanned) text.")

    parts = []
    if sections.get("Education"):
        parts.append("Computer Science / Data Science background")
    skill_phrase = ""
    if resume_skills:
        top_skills = resume_skills[:6]
        skill_phrase = "proficiency in " + ", ".join(s.title() for s in top_skills)

    sentence1 = "Motivated undergraduate"
    if parts:
        sentence1 += f" with a {parts[0]}"
    if skill_phrase:
        sentence1 += f", demonstrating {skill_phrase}."
    else:
        sentence1 += "."

    sentence2 = ""
    if sections.get("Projects") and sections.get("Experience"):
        sentence2 = " Backed by hands-on project work and practical experience."
    elif sections.get("Projects"):
        sentence2 = " Backed by hands-on project work."
    elif sections.get("Experience"):
        sentence2 = " Backed by practical work experience."

    sentence3 = ""
    if top_role:
        sentence3 = f" Skill profile shows strongest alignment with {top_role} roles."

    return sentence1 + sentence2 + sentence3


def generate_report(resume_skills, job_skills, matched, missing, ats, job_match_pct,
                     roles, sections, keyword_found, keyword_missing, strengths,
                     suggestions, summary) -> str:
    lines = []
    lines.append("AI RESUME ANALYZER & JOB MATCHER - ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("Created by Tibah Wajahat | CSE - Data Science")
    lines.append("=" * 60)

    lines.append(f"\nATS-STYLE SCORE: {ats['total']} / 100")
    lines.append(f"  - Skill match component:       {ats['skills_component']} / 40")
    lines.append(f"  - Resume completeness:         {ats['completeness_component']} / 25")
    lines.append(f"  - Keyword presence:            {ats['keyword_component']} / 20")
    lines.append(f"  - Content richness (length):   {ats['length_component']} / 15")

    lines.append(f"\nJOB MATCH: {job_match_pct}%")

    lines.append("\nBEST-FIT ROLES:")
    for role, pct in roles[:5]:
        lines.append(f"  - {role}: {pct}%")

    lines.append("\nMATCHED SKILLS:")
    lines.append("  " + (", ".join(s.title() for s in matched) if matched else "None detected"))

    lines.append("\nMISSING SKILLS (present in job description, not detected in resume):")
    lines.append("  " + (", ".join(s.title() for s in missing) if missing else "None"))

    lines.append("\nRESUME SECTIONS DETECTED:")
    for section, present in sections.items():
        lines.append(f"  - {section}: {'Yes' if present else 'Not detected'}")

    lines.append("\nKEYWORD ANALYSIS:")
    lines.append("  Found: " + (", ".join(keyword_found) if keyword_found else "None"))
    lines.append("  Missing: " + (", ".join(keyword_missing) if keyword_missing else "None"))

    lines.append("\nRESUME STRENGTHS:")
    for s in strengths:
        lines.append(f"  - {s}")

    lines.append("\nIMPROVEMENT SUGGESTIONS:")
    for s in suggestions:
        lines.append(f"  - {s}")

    lines.append("\nSUGGESTED PROFESSIONAL SUMMARY:")
    lines.append(f"  {summary}")

    lines.append("\n" + "=" * 60)
    lines.append("Note: All results are generated from the actual uploaded resume and job")
    lines.append("description using rule-based text analysis. No qualifications or skills")
    lines.append("were invented.")

    return "\n".join(lines)


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>📄 AI Resume Analyzer & Job Matcher</h1>
        <p>Created by Tibah Wajahat | CSE – Data Science</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload your resume and paste a job description to get an ATS-style score, "
    "job match percentage, skill gap analysis, best-fit role estimate, keyword "
    "analysis, and interview preparation — all computed from your actual inputs."
)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ About this App")
    st.write(
        """
        This tool analyzes a resume using rule-based text extraction and
        keyword/skill matching — no external AI API is used.

        **Features**
        - ATS-style score (0–100)
        - Job match percentage
        - Matched & missing skills
        - Best-fit job roles
        - Skill match chart
        - Keyword analysis
        - Resume section detection
        - Resume strengths
        - Interview question prep
        - Suggested professional summary
        - Downloadable report
        """
    )
    st.divider()
    st.caption("Tech stack: Python · Streamlit · PyPDF2 · Plotly · Pandas")
    st.caption("No data is stored — analysis happens only in this session.")

st.divider()

# =============================================================================
# INPUTS
# =============================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1️⃣ Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])

with col_right:
    st.subheader("2️⃣ Enter Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=220,
        placeholder="Example: We are looking for a Data Analyst with Python, SQL, "
                    "Excel and Power BI skills...",
    )

st.write("")
analyze_clicked = st.button("🔍 Analyze Resume", use_container_width=True)

# =============================================================================
# ANALYSIS
# =============================================================================
if analyze_clicked:

    if uploaded_file is None:
        st.error("Please upload your resume PDF.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text:
            st.error(
                "Could not extract any text from the uploaded PDF. It may be a "
                "scanned image or corrupted file. Please upload a text-based PDF."
            )
        else:
            with st.spinner("Crunching skills, keywords, and scores..."):
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)
                matched_skills = sorted(set(resume_skills).intersection(set(job_skills)))
                missing_skills = sorted(set(job_skills) - set(resume_skills))

                sections = detect_sections(resume_text)
                job_match_pct = calculate_job_match(resume_skills, job_skills)
                roles = best_fit_roles(resume_skills)

                keyword_found, keyword_missing = keyword_analysis(
                    resume_text, job_description, set(SKILLS)
                )
                ats = compute_ats_score(
                    resume_text, sections, resume_skills, job_skills,
                    keyword_found, len(keyword_found) + len(keyword_missing)
                )

                strengths = generate_strengths(resume_skills, sections, ats, job_match_pct)
                suggestions = generate_suggestions(missing_skills, sections, ats, keyword_missing)
                top_role = roles[0][0] if roles and roles[0][1] > 0 else ""
                summary = generate_summary(resume_skills, sections, top_role)

                report_text = generate_report(
                    resume_skills, job_skills, matched_skills, missing_skills, ats,
                    job_match_pct, roles, sections, keyword_found, keyword_missing,
                    strengths, suggestions, summary
                )

        if resume_text:
            st.success("Analysis completed successfully!")
            st.divider()

            # --- Summary metric cards ---
            st.subheader("📊 Overview")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="value">{ats["total"]}</div>'
                    f'<div class="label">ATS Score</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="value">{job_match_pct}%</div>'
                    f'<div class="label">Job Match</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="value">{len(resume_skills)}</div>'
                    f'<div class="label">Resume Skills</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                best_role_display = top_role if top_role else "N/A"
                st.markdown(
                    f'<div class="metric-card"><div class="value" style="font-size:1.25rem;">'
                    f'{best_role_display}</div><div class="label">Best-Fit Role</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("**ATS Score Breakdown**")
            b1, b2 = st.columns(2)
            with b1:
                st.caption(f"Skill Match — {ats['skills_component']} / 40")
                st.progress(min(ats["skills_component"] / 40, 1.0))
                st.caption(f"Resume Completeness — {ats['completeness_component']} / 25")
                st.progress(min(ats["completeness_component"] / 25, 1.0))
            with b2:
                st.caption(f"Keyword Presence — {ats['keyword_component']} / 20")
                st.progress(min(ats["keyword_component"] / 20, 1.0))
                st.caption(f"Content Richness — {ats['length_component']} / 15")
                st.progress(min(ats["length_component"] / 15, 1.0))

            st.divider()

            # --- Tabs for detailed results ---
            tab_skills, tab_roles, tab_sections, tab_keywords, tab_insights, tab_interview = st.tabs(
                ["🧩 Skills", "💼 Best-Fit Roles", "📄 Sections", "🔍 Keywords",
                 "🏆 Insights", "🎤 Interview Prep"]
            )

            with tab_skills:
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown("**✅ Matched Skills**")
                    if matched_skills:
                        st.markdown(
                            "".join(f'<span class="pill pill-matched">{s.title()}</span>'
                                    for s in matched_skills),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No matching skills were detected.")
                with sc2:
                    st.markdown("**⚠️ Missing Skills**")
                    if missing_skills:
                        st.markdown(
                            "".join(f'<span class="pill pill-missing">{s.title()}</span>'
                                    for s in missing_skills),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.success("No major missing skills detected!")

                st.write("")
                st.markdown("**Skill Match Chart**")
                if job_skills:
                    chart_df = pd.DataFrame({
                        "Skill": sorted(set(job_skills)),
                        "In Resume": [1 if s in resume_skills else 0 for s in sorted(set(job_skills))],
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=chart_df["Skill"].str.title(),
                        y=chart_df["In Resume"],
                        marker_color=["#4f46e5" if v == 1 else "#e5e7eb" for v in chart_df["In Resume"]],
                    ))
                    fig.update_layout(
                        yaxis=dict(tickvals=[0, 1], ticktext=["Missing", "Present"], range=[0, 1.2]),
                        height=380, margin=dict(t=20, b=20, l=10, r=10),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No predefined skills were detected in the job description.")

                with st.expander("🧠 All Skills Detected in Resume"):
                    st.write(", ".join(s.title() for s in resume_skills) if resume_skills else "None detected.")
                with st.expander("💼 All Skills Detected in Job Description"):
                    st.write(", ".join(s.title() for s in job_skills) if job_skills else "None detected.")

            with tab_roles:
                st.markdown("**Estimated compatibility with common roles**")
                for role, pct in roles:
                    st.caption(f"{role} — {pct}%")
                    st.progress(min(pct / 100, 1.0))

            with tab_sections:
                st.markdown("**Resume Section Detection**")
                sec_cols = st.columns(4)
                for i, (section, present) in enumerate(sections.items()):
                    with sec_cols[i % 4]:
                        icon = "✅" if present else "❌"
                        st.write(f"{icon} {section}")

            with tab_keywords:
                kc1, kc2 = st.columns(2)
                with kc1:
                    st.markdown("**Found in Resume**")
                    if keyword_found:
                        st.markdown(
                            "".join(f'<span class="pill pill-matched">{k}</span>' for k in keyword_found),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No additional keywords detected.")
                with kc2:
                    st.markdown("**Not Found in Resume**")
                    if keyword_missing:
                        st.markdown(
                            "".join(f'<span class="pill pill-missing">{k}</span>' for k in keyword_missing),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.success("No major keyword gaps detected!")

            with tab_insights:
                st.markdown("**🏆 Resume Strengths**")
                for s in strengths:
                    st.write(f"- {s}")
                st.write("")
                st.markdown("**📝 Improvement Suggestions**")
                for s in suggestions:
                    st.write(f"- {s}")
                st.write("")
                st.markdown("**✨ Suggested Professional Summary**")
                st.info(summary)

            with tab_interview:
                st.markdown("**Technical Questions (based on detected & required skills)**")
                relevant_skills = sorted(set(resume_skills) | set(job_skills))
                any_q = False
                for skill in relevant_skills:
                    if skill in INTERVIEW_QUESTIONS:
                        any_q = True
                        with st.expander(f"❓ {skill.title()}"):
                            for q in INTERVIEW_QUESTIONS[skill]:
                                st.write(f"- {q}")
                if not any_q:
                    st.info("No technical question bank matched the detected skills.")
                st.write("")
                st.markdown("**General HR Questions**")
                for q in HR_QUESTIONS:
                    st.write(f"- {q}")

            st.divider()

            # --- Download report ---
            st.subheader("📥 Download Report")
            st.download_button(
                label="Download Full Analysis Report (.md)",
                data=report_text,
                file_name="resume_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

            with st.expander("📄 View Extracted Resume Text"):
                st.text_area("Resume Content", resume_text, height=300)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    f"""
    <div class="footer">
        © {datetime.now().year} AI Resume Analyzer | Created by Tibah Wajahat | CSE – Data Science
    </div>
    """,
    unsafe_allow_html=True,
)