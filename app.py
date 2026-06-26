import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import re
import hashlib
import io
from datetime import datetime, date, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader
from docx import Document
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Enterprise AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_NAME = "ats_enterprise.db"

# =========================================================
# UTILITIES (DEFINE BEFORE DB INIT)
# =========================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def safe_float(v, default=0):
    try:
        return float(v)
    except:
        return default

# =========================================================
# SESSION STATE
# =========================================================
def init_session():
    defaults = {
        "logged_in": False,
        "user_name": "",
        "user_email": "",
        "user_role": "",
        "user_department": "General",
        "theme_mode": "Light",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =========================================================
# DATABASE
# =========================================================
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False, timeout=30)

def execute_query(query, params=(), fetch=False, many=False):
    conn = get_connection()
    cur = conn.cursor()
    try:
        if many:
            cur.executemany(query, params)
        else:
            cur.execute(query, params)
        conn.commit()
        if fetch:
            return cur.fetchall()
    finally:
        conn.close()

def get_df(query):
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT,
        department TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT,
        department TEXT,
        required_skills TEXT,
        experience_required TEXT,
        job_description TEXT,
        created_by TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT,
        candidate_email TEXT,
        phone TEXT,
        education TEXT,
        experience TEXT,
        skills TEXT,
        resume_text TEXT,
        job_role TEXT,
        score REAL,
        skill_score REAL,
        experience_score REAL,
        communication_score REAL,
        fake_resume_score REAL,
        confidence_level TEXT,
        recommendation TEXT,
        skill_gap TEXT,
        status TEXT,
        interview_status TEXT,
        applied_on TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS interviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT,
        candidate_email TEXT,
        job_role TEXT,
        interview_date TEXT,
        interview_time TEXT,
        mode TEXT,
        interviewer TEXT,
        status TEXT,
        feedback TEXT,
        created_at TEXT
    )
    """)

    conn.commit()

    cur.execute("SELECT * FROM users WHERE email=?", ("admin@ats.com",))
    admin = cur.fetchone()
    if not admin:
        cur.execute("""
        INSERT INTO users (full_name, email, password, role, department, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, ("System Admin", "admin@ats.com", hash_password("admin123"), "Admin", "General", now()))
        conn.commit()

    conn.close()

init_db()

# =========================================================
# DATA HELPERS
# =========================================================
def insert_user(full_name, email, password, role, department):
    try:
        execute_query("""
        INSERT INTO users (full_name, email, password, role, department, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, email, hash_password(password), role, department, now()))
        return True
    except:
        return False

def login_user(email, password):
    rows = execute_query(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, hash_password(password)),
        fetch=True
    )
    return rows[0] if rows else None

def get_users_df():
    return get_df("SELECT * FROM users ORDER BY id DESC")

def get_jobs_df():
    return get_df("SELECT * FROM jobs ORDER BY id DESC")

def get_candidates_df():
    return get_df("SELECT * FROM candidates ORDER BY id DESC")

def get_interviews_df():
    return get_df("SELECT * FROM interviews ORDER BY id DESC")

# =========================================================
# THEME / UI
# =========================================================
def apply_theme():
    if st.session_state.theme_mode == "Dark":
        bg = "#07111f"
        bg2 = "#0b1220"
        glass = "rgba(17, 24, 39, 0.72)"
        glass_soft = "rgba(30, 41, 59, 0.58)"
        text = "#f8fafc"
        muted = "#cbd5e1"
        border = "rgba(255,255,255,0.08)"
        sidebar_bg = "rgba(9, 14, 26, 0.88)"
        chip_bg = "rgba(59,130,246,0.14)"
    else:
        bg = "#eef5ff"
        bg2 = "#f8fbff"
        glass = "rgba(255, 255, 255, 0.68)"
        glass_soft = "rgba(255, 255, 255, 0.56)"
        text = "#0f172a"
        muted = "#475569"
        border = "rgba(15, 23, 42, 0.08)"
        sidebar_bg = "rgba(255, 255, 255, 0.82)"
        chip_bg = "rgba(37,99,235,0.10)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 12% 10%, rgba(59,130,246,0.18), transparent 22%),
            radial-gradient(circle at 85% 8%, rgba(168,85,247,0.14), transparent 24%),
            radial-gradient(circle at 78% 78%, rgba(14,165,233,0.12), transparent 26%),
            linear-gradient(135deg, {bg}, {bg2});
        color: {text};
    }}

    .block-container {{
        padding-top: 6.2rem !important;
        padding-bottom: 6rem !important;
        max-width: 100% !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    .brand-card {{
        padding: 18px 16px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.18));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 10px 30px rgba(0,0,0,0.10);
        backdrop-filter: blur(16px);
        margin-bottom: 14px;
    }}

    .brand-title {{
        font-size: 24px;
        font-weight: 900;
        line-height: 1.2;
    }}

    .brand-sub {{
        font-size: 13px;
        color: {muted};
        margin-top: 4px;
    }}

    .hero-card {{
        width: 100%;
        padding: 28px 28px;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.14)),
            {glass};
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 12px 36px rgba(15, 23, 42, 0.10);
        backdrop-filter: blur(18px);
        margin-bottom: 24px;
        overflow: visible !important;
    }}

    .hero-title {{
        font-size: 40px;
        font-weight: 900;
        line-height: 1.15;
        color: {text};
        margin-bottom: 8px;
        word-break: break-word;
        white-space: normal !important;
    }}

    .hero-subtitle {{
        font-size: 15px;
        color: {muted};
        line-height: 1.75;
        max-width: 980px;
    }}

    .section-title {{
        font-size: 27px;
        font-weight: 800;
        color: {text};
        margin-top: 10px;
        margin-bottom: 4px;
    }}

    .section-sub {{
        font-size: 14px;
        color: {muted};
        margin-bottom: 18px;
    }}

    .glass-card {{
        background: {glass};
        border: 1px solid {border};
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(18px);
        margin-bottom: 18px;
    }}

    .auth-card {{
        background: {glass};
        border: 1px solid {border};
        border-radius: 28px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        backdrop-filter: blur(18px);
    }}

    [data-testid="stMetric"] {{
        background: {glass};
        border: 1px solid {border};
        border-radius: 22px;
        padding: 18px 14px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
        backdrop-filter: blur(18px);
    }}

    .stButton > button {{
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 800 !important;
        border: 1px solid rgba(59,130,246,0.20) !important;
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
        box-shadow: 0 10px 24px rgba(37,99,235,0.24);
    }}

    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stDateInput input, .stTimeInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 15px !important;
    }}

    .skill-chip {{
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: {chip_bg};
        border: 1px solid rgba(59,130,246,0.18);
        margin: 4px 6px 4px 0;
        font-size: 12px;
        font-weight: 700;
        color: {text};
    }}

    .pretty-divider {{
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, transparent, rgba(100,116,139,0.35), transparent);
        margin: 16px 0 20px 0;
    }}

    h1, h2, h3, h4, h5, h6 {{
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        white-space: normal !important;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

def render_hero(title, subtitle, badge="AI Powered"):
    st.markdown(f"""
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:20px; flex-wrap:wrap;">
            <div>
                <div class="hero-title">{title}</div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <div style="
                padding:8px 14px;
                border-radius:999px;
                background:rgba(255,255,255,0.18);
                border:1px solid rgba(255,255,255,0.16);
                font-size:12px;
                font-weight:800;
                white-space:nowrap;
            ">
                ✨ {badge}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_section(title, subtitle=""):
    st.markdown(f"""
    <div class="section-title">{title}</div>
    <div class="section-sub">{subtitle}</div>
    """, unsafe_allow_html=True)

def render_brand():
    st.markdown("""
    <div class="brand-card">
        <div class="brand-title">📄 Enterprise ATS</div>
        <div class="brand-sub">AI Resume Screening System</div>
    </div>
    """, unsafe_allow_html=True)

def render_divider():
    st.markdown('<div class="pretty-divider"></div>', unsafe_allow_html=True)

def render_skill_chips(skills_text):
    if not skills_text:
        return
    skills = [s.strip() for s in str(skills_text).split(",") if s.strip()]
    html = "".join([f'<span class="skill-chip">{skill}</span>' for skill in skills[:15]])
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# NLP / AI ENGINE
# =========================================================
COMMON_SKILLS = [
    "python", "java", "sql", "machine learning", "deep learning", "nlp", "power bi",
    "excel", "communication", "teamwork", "leadership", "react", "node.js", "mongodb",
    "html", "css", "javascript", "django", "flask", "streamlit", "tableau", "aws",
    "azure", "docker", "git", "linux", "data analysis", "pandas", "numpy", "scikit-learn"
]

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except:
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except:
        return ""

def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else ""

def extract_phone(text):
    match = re.search(r'(\+91[\-\s]?)?[6-9]\d{9}', text)
    return match.group(0) if match else ""

def extract_name(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return lines[0][:50]
    return "Unknown Candidate"

def extract_skills(text):

    skill_keywords = [
        "python", "java", "sql", "machine learning", "deep learning",
        "nlp", "power bi", "excel", "communication", "teamwork",
        "leadership", "react", "node.js", "mongodb", "html",
        "css", "javascript", "django", "flask", "streamlit",
        "tableau", "aws", "azure", "docker", "git", "linux",
        "data analysis", "pandas", "numpy", "scikit-learn",
        "tensorflow", "keras", "c++", "php", "bootstrap"
    ]

    found_skills = []

    text = text.lower()

    for skill in skill_keywords:
        if skill.lower() in text:
            found_skills.append(skill.title())

    return list(set(found_skills))

def score_communication(text):
    words = re.findall(r'\w+', text)
    unique = len(set(words))
    total = len(words)
    if total == 0:
        return 0
    score = min(100, int((unique / total) * 300))
    return max(score, 40)

def score_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs|year)', text.lower())
    if matches:
        years = max([int(m[0]) for m in matches])
        return min(100, years * 12)
    return np.random.randint(40, 75)

def detect_fake_resume(text):
    suspicious_words = ["lorem ipsum", "dummy", "sample", "template", "copy paste", "fake", "generated"]
    count = sum(1 for w in suspicious_words if w in text.lower())
    score = min(100, count * 18 + np.random.randint(5, 25))
    return score

def compute_tfidf_similarity(resume_text, jd_text):
    docs = [resume_text, jd_text]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(docs)
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(sim * 100, 2)
    except:
        return 0.0

def analyze_resume(resume_text, job_title, jd_skills, jd_description):
    candidate_name = extract_name(resume_text)
    candidate_email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    extracted_skills = extract_skills(resume_text)

    jd_skill_list = [s.strip().lower() for s in str(jd_skills).split(",") if s.strip()]
    candidate_skill_list = [s.lower() for s in extracted_skills]

    matched_skills = [s for s in jd_skill_list if s in candidate_skill_list]
    missing_skills = [s.title() for s in jd_skill_list if s not in candidate_skill_list]

    skill_score = round((len(matched_skills) / max(1, len(jd_skill_list))) * 100, 2)
    exp_score = score_experience(resume_text)
    comm_score = score_communication(resume_text)
    fake_score = detect_fake_resume(resume_text)

    jd_text = f"{job_title} {jd_skills} {jd_description}"
    tfidf_score = compute_tfidf_similarity(resume_text, jd_text)

    final_score = round((skill_score * 0.35) + (exp_score * 0.20) + (comm_score * 0.15) + (tfidf_score * 0.30), 2)

    if final_score >= 65:
        confidence = "High"
        recommendation = "Strongly Recommended"
        status = "Shortlisted"

    elif final_score >= 45:
        confidence = "Medium"
        recommendation = "Recommended"
        status = "Under Review"

    else:
        confidence = "Low"
        recommendation = "Not Recommended"
        status = "Rejected"

    return {
        "candidate_name": candidate_name,
        "candidate_email": candidate_email if candidate_email else "not_found@email.com",
        "phone": phone if phone else "Not Found",
        "education": "Graduate",
        "experience": f"{round(exp_score / 12, 1)} Years",
        "skills": ", ".join(extracted_skills),
        "resume_text": resume_text,
        "job_role": job_title,
        "score": final_score,
        "skill_score": skill_score,
        "experience_score": exp_score,
        "communication_score": comm_score,
        "fake_resume_score": fake_score,
        "confidence_level": confidence,
        "recommendation": recommendation,
        "skill_gap": ", ".join(missing_skills),
        "status": status,
        "interview_status": "Pending",
        "applied_on": now()
    }

def save_candidate(candidate_data):
    execute_query("""
    INSERT INTO candidates (
        candidate_name, candidate_email, phone, education, experience, skills, resume_text,
        job_role, score, skill_score, experience_score, communication_score, fake_resume_score,
        confidence_level, recommendation, skill_gap, status, interview_status, applied_on
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        candidate_data["candidate_name"],
        candidate_data["candidate_email"],
        candidate_data["phone"],
        candidate_data["education"],
        candidate_data["experience"],
        candidate_data["skills"],
        candidate_data["resume_text"],
        candidate_data["job_role"],
        candidate_data["score"],
        candidate_data["skill_score"],
        candidate_data["experience_score"],
        candidate_data["communication_score"],
        candidate_data["fake_resume_score"],
        candidate_data["confidence_level"],
        candidate_data["recommendation"],
        candidate_data["skill_gap"],
        candidate_data["status"],
        candidate_data["interview_status"],
        candidate_data["applied_on"]
    ))

# =========================================================
# PDF REPORT
# =========================================================
def create_candidate_pdf(candidate):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Candidate AI Screening Report")

    c.setFont("Helvetica", 11)
    y = 770
    lines = [
        f"Candidate Name: {candidate['candidate_name']}",
        f"Email: {candidate['candidate_email']}",
        f"Phone: {candidate['phone']}",
        f"Applied Role: {candidate['job_role']}",
        f"Final Score: {candidate['score']}%",
        f"Skill Match: {candidate['skill_score']}%",
        f"Experience Score: {candidate['experience_score']}%",
        f"Communication Score: {candidate['communication_score']}%",
        f"Fake Resume Risk: {candidate['fake_resume_score']}%",
        f"Confidence Level: {candidate['confidence_level']}",
        f"Recommendation: {candidate['recommendation']}",
        f"Skills: {candidate['skills']}",
        f"Skill Gap: {candidate['skill_gap']}",
        f"Status: {candidate['status']}",
        f"Interview Status: {candidate['interview_status']}",
        f"Applied On: {candidate['applied_on']}",
    ]

    for line in lines:
        c.drawString(50, y, line[:120])
        y -= 22

    c.save()
    buffer.seek(0)
    return buffer

# =========================================================
# AUTH PAGE
# =========================================================
def auth_page():
    render_hero(
        "📄 Enterprise AI Resume Screening System",
        "Professional NLP-based Applicant Tracking System with AI Screening, Rankings, Interview Workflow, Reports, and Hiring Intelligence",
        "Enterprise Ready"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Modules", "12+")
    c2.metric("User Roles", "4")
    c3.metric("Analytics", "Live")
    c4.metric("Project Type", "Enterprise")

    render_divider()

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        render_section("🚀 Why This Platform?", "Built like a modern ATS product for real-world hiring simulation")
        st.markdown("""
- AI Resume Screening  
- Candidate Ranking & Matching  
- Skill Gap Analysis  
- Interview Scheduling  
- Recruiter AI Assistant  
- Admin + Recruiter + Candidate Roles  
- Exportable Reports & Hiring Analytics  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

        with tab1:
            st.markdown("### Welcome Back")
            email = st.text_input("Email Address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", use_container_width=True):
                user = login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_name = user[1]
                    st.session_state.user_email = user[2]
                    st.session_state.user_role = user[4]
                    st.session_state.user_department = user[5] if user[5] else "General"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

            st.info("Default Admin Login → admin@ats.com / admin123")

        with tab2:
            st.markdown("### Create New Account")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            password = st.text_input("Create Password", type="password")
            role = st.selectbox("Role", ["Recruiter", "HR Manager", "Admin", "Candidate"])
            department = st.selectbox("Department", ["General", "HR", "IT", "Data Science", "Operations", "Finance", "Marketing"])

            if st.button("Create Account", use_container_width=True):
                if full_name and email and password:
                    success = insert_user(full_name, email, password, role, department)
                    if success:
                        st.success("Registration successful. Please login.")
                    else:
                        st.error("Email already exists.")
                else:
                    st.warning("Please fill all fields.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# APP PAGES
# =========================================================
def dashboard_page():
    render_hero(
        f"👋 Welcome, {st.session_state.user_name}",
        "Enterprise Hiring Intelligence Dashboard with live metrics, rankings, analytics, and workflow tracking",
        "Live Dashboard"
    )

    cand_df = get_candidates_df()
    jobs_df = get_jobs_df()
    int_df = get_interviews_df()

    total_candidates = len(cand_df)
    shortlisted = len(cand_df[cand_df["status"] == "Shortlisted"]) if not cand_df.empty else 0
    total_jobs = len(jobs_df)
    interviews = len(int_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", total_candidates)
    c2.metric("Shortlisted", shortlisted)
    c3.metric("Open Jobs", total_jobs)
    c4.metric("Interviews", interviews)

    render_divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Candidate Status Distribution")
        if not cand_df.empty:
            fig = px.pie(cand_df, names="status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candidate data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Top Candidate Scores")
        if not cand_df.empty:
            top_df = cand_df.sort_values("score", ascending=False).head(10)
            fig = px.bar(top_df, x="candidate_name", y="score", text="score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candidate data available.")
        st.markdown('</div>', unsafe_allow_html=True)

def admin_panel():
    render_hero("🛡️ Admin Control Center", "Manage users and platform visibility", "Admin Access")
    users_df = get_users_df()
    st.dataframe(users_df, use_container_width=True)

def job_management():
    render_hero("💼 Job Management", "Create and manage enterprise hiring openings", "Hiring Ops")

    with st.form("job_form"):
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title")
            department = st.selectbox("Department", ["IT", "HR", "Data Science", "Operations", "Finance", "Marketing", "General"])
            experience_required = st.text_input("Experience Required")
        with col2:
            required_skills = st.text_area("Required Skills (comma separated)", height=100)
            job_description = st.text_area("Job Description", height=100)

        submitted = st.form_submit_button("Create Job")
        if submitted:
            if job_title and required_skills:
                execute_query("""
                INSERT INTO jobs (job_title, department, required_skills, experience_required, job_description, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_title, department, required_skills, experience_required, job_description,
                    st.session_state.user_email, now()
                ))
                st.success("Job created successfully.")
                st.rerun()
            else:
                st.warning("Please fill required fields.")

    st.dataframe(get_jobs_df(), use_container_width=True)

def applicant_portal():
    render_hero("🧑‍🎓 Applicant Self-Application Portal", "Real-time candidate application experience", "Candidate Experience")

    jobs_df = get_jobs_df()
    if jobs_df.empty:
        st.warning("No jobs available yet. Please add jobs first.")
        return

    selected_job = st.selectbox("Select Job Role", jobs_df["job_title"].tolist())
    selected_job_row = jobs_df[jobs_df["job_title"] == selected_job].iloc[0]

    candidate_name = st.text_input("Candidate Full Name")
    candidate_email = st.text_input("Candidate Email")
    phone = st.text_input("Phone Number")
    education = st.text_input("Education")
    experience = st.text_input("Experience")
    manual_skills = st.text_input("Skills (comma separated)")
    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if st.button("Submit Application", use_container_width=True):
        if uploaded_resume and candidate_name and candidate_email:
            resume_text = extract_resume_text(uploaded_resume)
            analysis = analyze_resume(
                resume_text=resume_text + f"\n{manual_skills}\n{experience}\n{education}",
                job_title=selected_job_row["job_title"],
                jd_skills=selected_job_row["required_skills"],
                jd_description=selected_job_row["job_description"]
            )

            analysis["candidate_name"] = candidate_name
            analysis["candidate_email"] = candidate_email
            analysis["phone"] = phone if phone else analysis["phone"]
            analysis["education"] = education if education else "Graduate"
            analysis["experience"] = experience if experience else analysis["experience"]

            save_candidate(analysis)
            st.success("Application submitted successfully.")
            st.balloons()
        else:
            st.warning("Please fill required fields and upload resume.")

def screening_lab():
    render_hero("📤 Resume Screening Lab", "Batch AI resume screening engine", "AI Screening")

    jobs_df = get_jobs_df()
    if jobs_df.empty:
        st.warning("Please create at least one job first.")
        return

    selected_job = st.selectbox("Select Job for Screening", jobs_df["job_title"].tolist())
    job_row = jobs_df[jobs_df["job_title"] == selected_job].iloc[0]

    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("Run AI Screening", use_container_width=True):
        if uploaded_files:
            results = []
            for file in uploaded_files:
                resume_text = extract_resume_text(file)
                analysis = analyze_resume(
                    resume_text=resume_text,
                    job_title=job_row["job_title"],
                    jd_skills=job_row["required_skills"],
                    jd_description=job_row["job_description"]
                )
                save_candidate(analysis)
                results.append(analysis)

            results_df = pd.DataFrame(results).sort_values("score", ascending=False)
            st.success("AI Screening Completed Successfully.")
            st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("Please upload resumes first.")

def candidate_pipeline():
    render_hero("📋 Candidate Pipeline", "Track, shortlist, compare, and manage candidates", "ATS Workflow")

    df = get_candidates_df()
    if df.empty:
        st.info("No candidates found yet.")
        return

    status_filter = st.selectbox("Filter by Status", ["All"] + sorted(df["status"].dropna().unique().tolist()))
    if status_filter != "All":
        df = df[df["status"] == status_filter]

    for _, row in df.head(12).iterrows():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2.4, 1, 1])

        with col1:
            st.markdown(f"### 👤 {row['candidate_name']}")
            st.caption(f"{row['candidate_email']} | {row['phone']}")
            st.write(f"**Applied Role:** {row['job_role']}")
            st.write(f"**Education:** {row['education']}")
            st.write(f"**Experience:** {row['experience']}")
            st.write("**Skills:**")
            render_skill_chips(row["skills"])

        with col2:
            st.metric("Final Score", f"{row['score']}%")
            st.metric("Skill Match", f"{row['skill_score']}%")
            st.metric("Confidence", row['confidence_level'])

        with col3:
            st.metric("Status", row['status'])
            st.metric("Interview", row['interview_status'])
            st.metric("Risk Score", f"{row['fake_resume_score']}%")

        pdf_buffer = create_candidate_pdf(row)
        st.download_button(
    label=f"⬇ Download Report - {row['candidate_name']}",
    data=pdf_buffer,
    file_name=f"{row['candidate_name']}_report.pdf",
    mime="application/pdf",
    key=f"download_report_{row['id']}"
)
        st.markdown('</div>', unsafe_allow_html=True)

def interview_scheduling():
    render_hero(
        "📅 Interview Scheduling",
        "Plan and monitor candidate interviews",
        "Interview Flow"
    )

    cand_df = get_candidates_df()

    # =====================================================
    # FILTER OUT REJECTED CANDIDATES
    # =====================================================
    eligible_candidates = cand_df[
        cand_df["status"].isin(["Shortlisted", "Under Review"])
    ]

    if eligible_candidates.empty:
        st.info("No eligible candidates available for interview scheduling.")
        return

    selected_candidate = st.selectbox(
        "Select Candidate",
        eligible_candidates["candidate_name"].tolist()
    )

    cand = eligible_candidates[
        eligible_candidates["candidate_name"] == selected_candidate
    ].iloc[0]

    with st.form("interview_form"):

        col1, col2 = st.columns(2)

        with col1:
            interview_date = st.date_input(
                "Interview Date",
                value=date.today()
            )

            mode = st.selectbox(
                "Mode",
                ["Online", "Offline", "Telephonic"]
            )

        with col2:
            interview_time = st.time_input(
                "Interview Time",
                value=time(10, 0)
            )

            interviewer = st.text_input("Interviewer Name")

        submit = st.form_submit_button("Schedule Interview")

        if submit:

            execute_query("""
            INSERT INTO interviews (
                candidate_name,
                candidate_email,
                job_role,
                interview_date,
                interview_time,
                mode,
                interviewer,
                status,
                feedback,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cand["candidate_name"],
                cand["candidate_email"],
                cand["job_role"],
                str(interview_date),
                str(interview_time),
                mode,
                interviewer,
                "Scheduled",
                "",
                now()
            ))

            execute_query("""
            UPDATE candidates
            SET interview_status=?
            WHERE candidate_email=?
            """, (
                "Scheduled",
                cand["candidate_email"]
            ))

            st.success("Interview scheduled successfully.")
            st.rerun()

    st.dataframe(get_interviews_df(), use_container_width=True)

def ai_insights():
    render_hero("🤖 AI Insights & Hiring Intelligence", "Advanced explainability and fit recommendation", "Explainable AI")
    df = get_candidates_df()
    if df.empty:
        st.info("No candidate data available.")
        return

    top_df = df.sort_values("score", ascending=False).head(10)
    st.dataframe(top_df[[
        "candidate_name", "job_role", "score", "skill_score",
        "experience_score", "communication_score", "confidence_level", "recommendation"
    ]], use_container_width=True)

def reports_center():
    render_hero("📊 Reports Center", "Generate export-ready hiring reports", "Reporting Suite")

    cand_df = get_candidates_df()
    int_df = get_interviews_df()
    jobs_df = get_jobs_df()

    tab1, tab2, tab3 = st.tabs(["📄 Candidates", "📅 Interviews", "💼 Jobs"])

    with tab1:
        st.dataframe(cand_df, use_container_width=True)
        if not cand_df.empty:
            st.download_button("⬇ Download Candidates CSV", cand_df.to_csv(index=False).encode("utf-8"), "candidates_report.csv", "text/csv")

    with tab2:
        st.dataframe(int_df, use_container_width=True)
        if not int_df.empty:
            st.download_button("⬇ Download Interviews CSV", int_df.to_csv(index=False).encode("utf-8"), "interviews_report.csv", "text/csv")

    with tab3:
        st.dataframe(jobs_df, use_container_width=True)
        if not jobs_df.empty:
            st.download_button("⬇ Download Jobs CSV", jobs_df.to_csv(index=False).encode("utf-8"), "jobs_report.csv", "text/csv")

def recruiter_ai_assistant():
    render_hero("💬 Recruiter AI Assistant", "Smart hiring support assistant", "Assistant Mode")

    question = st.text_area("Ask your recruiter assistant", placeholder="Example: Who are the top candidates for Data Scientist role?")

    if st.button("Get AI Suggestion", use_container_width=True):
        df = get_candidates_df()

        if df.empty:
            st.warning("No candidate data available.")
        else:
            q = question.lower()

            if "top candidate" in q or "best candidate" in q:
                top = df.sort_values("score", ascending=False).head(3)
                st.success("Top AI Recommended Candidates")
                st.dataframe(top[["candidate_name", "job_role", "score", "recommendation"]], use_container_width=True)

            elif "skill gap" in q:
                st.dataframe(df[["candidate_name", "job_role", "skill_gap"]], use_container_width=True)

            else:
                st.info("""
### AI Assistant Suggestion
- Review top scoring candidates first
- Focus on candidates with **High Confidence**
- Avoid candidates with very high **Fake Resume Risk**
- Prioritize candidates with lower **Skill Gap**
- Schedule interviews for shortlisted profiles
                """)

def settings_page():
    render_hero("⚙️ Settings", "Customize your ATS experience", "Configuration")
    mode = st.selectbox("Theme Mode", ["Light", "Dark"], index=0 if st.session_state.theme_mode == "Light" else 1)
    if mode != st.session_state.theme_mode:
        st.session_state.theme_mode = mode
        st.rerun()

def about_project():
    render_hero("ℹ️ About Project", "Final Year Enterprise AI Major Project with NLP", "Academic Ready")

    st.markdown("""
### **Project Title**
**Enterprise AI Resume Screening System using NLP**

### **Objective**
To automate resume screening, candidate ranking, interview scheduling, and hiring analytics using AI and NLP.

### **Core AI Features**
- Resume Parsing
- TF-IDF Resume Matching
- Skill Match Engine
- Experience Scoring
- Communication Scoring
- Fake Resume Detection
- Skill Gap Analysis
- Candidate Fit Recommendation
    """)

# =========================================================
# MAIN APP
# =========================================================
if not st.session_state.logged_in:
    auth_page()
else:
    with st.sidebar:
        render_brand()

        st.markdown("### 🧑 User Profile")
        st.success(st.session_state.user_name)
        st.caption(st.session_state.user_email)
        st.write(f"**Role:** {st.session_state.user_role}")
        st.write(f"**Dept:** {st.session_state.user_department}")

        render_divider()

        st.markdown("### 🎨 Theme")
        theme_choice = st.selectbox("Select Theme", ["Light", "Dark"], index=0 if st.session_state.theme_mode == "Light" else 1)
        if theme_choice != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_choice
            st.rerun()

        render_divider()

        page = st.radio("Navigation", [
            "Dashboard",
            "Admin Panel",
            "Job Management",
            "Applicant Portal",
            "Resume Screening Lab",
            "Candidate Pipeline",
            "Interview Scheduling",
            "AI Insights",
            "Reports Center",
            "Recruiter AI Assistant",
            "Settings",
            "About Project"
        ])

        render_divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            st.session_state.user_role = ""
            st.session_state.user_department = "General"
            st.rerun()

    if page == "Dashboard":
        dashboard_page()
    elif page == "Admin Panel":
        admin_panel()
    elif page == "Job Management":
        job_management()
    elif page == "Applicant Portal":
        applicant_portal()
    elif page == "Resume Screening Lab":
        screening_lab()
    elif page == "Candidate Pipeline":
        candidate_pipeline()
    elif page == "Interview Scheduling":
        interview_scheduling()
    elif page == "AI Insights":
        ai_insights()
    elif page == "Reports Center":
        reports_center()
    elif page == "Recruiter AI Assistant":
        recruiter_ai_assistant()
    elif page == "Settings":
        settings_page()
    elif page == "About Project":
        about_project()