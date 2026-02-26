import streamlit as st

st.set_page_config(
    page_title="Data Visualization Portfolio",
    page_icon="🚀",
    layout="wide",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-top: 0;
    }
    .hero-intro {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #ccc;
        max-width: 800px;
        margin: 16px 0 8px 0;
    }
    .week-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border-left: 4px solid;
        transition: transform 0.2s;
    }
    .week-card:hover {
        transform: translateX(6px);
    }
    .week-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.6;
    }
    .week-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .week-desc {
        font-size: 0.95rem;
        opacity: 0.8;
    }
    .level-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .divider-line {
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, transparent);
        border: none;
        margin: 32px 0 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ──
st.markdown('<p class="hero-title">From Zero to Hero 🚀</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">A 6-week journey through Data Science — from cloud setup to machine learning</p>', unsafe_allow_html=True)

st.markdown("""
<p class="hero-intro">
This portfolio documents my hands-on progression through a Data Visualization course at OAMK 🇫🇮.
Each week introduced a new layer of complexity — starting with 🌱 <strong>setting up a cloud environment and plotting population data</strong>,
then diving into 📈 <strong>stock market time series</strong> with yfinance and interactive Plotly charts.
From there, I explored 🔗 <strong>multi-stock correlation analysis</strong> using heatmaps, moved on to 🧪 <strong>statistical hypothesis testing</strong>
(t-tests, chi-square, confidence intervals), and built my first 🤖 <strong>supervised ML models</strong> — Random Forest, Logistic Regression, and linear regression for prediction and classification.
The journey culminates in 🦸 <strong>unsupervised learning</strong> — K-Means clustering, hierarchical analysis, and Factor Analysis to uncover hidden patterns in data.
Built entirely with Python, Streamlit, Pandas, and Scikit-learn.
</p>
""", unsafe_allow_html=True)

# ── Progress Bar ──
st.markdown("---")
col_prog, col_stat = st.columns([3, 1])
with col_prog:
    st.progress(1.0, text="✅ Journey Complete — 6/6 weeks finished")
with col_stat:
    st.metric("Skill Level", "🦸 Hero", delta="↑ from Zero")

st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# ── The Journey ──
st.markdown("### 🗺️ The Roadmap")
st.caption("Click each week in the sidebar to explore the full interactive analysis")

# Week data: (emoji, title, subtitle, skills, level_label, level_color, border_color)
weeks = [
    {
        "num": 1,
        "emoji": "🌱",
        "title": "Cloud Setup & Population Data",
        "desc": "First steps — setting up the cloud environment and visualizing real-world population trends.",
        "skills": ["Python", "Streamlit", "Cloud Deployment", "Bar Charts"],
        "level": "Seedling",
        "color": "#4ade80",
    },
    {
        "num": 2,
        "emoji": "📈",
        "title": "Time Series Analysis",
        "desc": "Learning to read the story behind stock price movements over time.",
        "skills": ["Pandas", "Time Series", "Line Charts", "Moving Averages"],
        "level": "Explorer",
        "color": "#60a5fa",
    },
    {
        "num": 3,
        "emoji": "🔗",
        "title": "Correlation Analysis",
        "desc": "Discovering hidden relationships between multiple stocks using heatmaps and scatter analysis.",
        "skills": ["Correlation Matrices", "Heatmaps", "Multi-Stock Analysis"],
        "level": "Analyst",
        "color": "#a78bfa",
    },
    {
        "num": 4,
        "emoji": "🧪",
        "title": "Statistical Hypothesis Testing",
        "desc": "Putting data under the microscope — t-tests, chi-square, confidence intervals, and effect sizes.",
        "skills": ["T-tests", "Chi-Square", "Cohen's d", "Confidence Intervals"],
        "level": "Scientist",
        "color": "#f472b6",
    },
    {
        "num": 5,
        "emoji": "🤖",
        "title": "Supervised Machine Learning",
        "desc": "Teaching machines to predict — regression for numbers, classification for categories.",
        "skills": ["Linear Regression", "Random Forest", "Logistic Regression", "Model Evaluation"],
        "level": "Engineer",
        "color": "#fb923c",
    },
    {
        "num": 6,
        "emoji": "🦸",
        "title": "Unsupervised Machine Learning",
        "desc": "The final boss — letting algorithms discover patterns on their own through clustering and factor analysis.",
        "skills": ["K-Means", "DBSCAN", "PCA", "Factor Analysis"],
        "level": "Hero",
        "color": "#facc15",
    },
]

# ── Render week cards ──
for i, w in enumerate(weeks):
    col1, col2 = st.columns([1, 15])
    
    with col1:
        st.markdown(f"<div style='font-size: 2.2rem; text-align: center; padding-top: 12px;'>{w['emoji']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            f"""
            <div class="week-card" style="border-left-color: {w['color']};">
                <span class="week-label" style="color: {w['color']};">Week {w['num']} · Level: {w['level']}</span>
                <div class="week-title">{w['title']}</div>
                <div class="week-desc">{w['desc']}</div>
                <div style="margin-top: 10px;">
                    {''.join(f'<span class="level-badge" style="background: {w["color"]}22; color: {w["color"]}; margin-right: 6px;">{s}</span>' for s in w['skills'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Arrow connector between weeks
    if i < len(weeks) - 1:
        st.markdown(
            f"<div style='text-align: center; font-size: 1.2rem; opacity: 0.3; margin: -4px 0;'>⬇</div>",
            unsafe_allow_html=True,
        )

# ── Footer ──
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**🛠️ Built With**")
    st.caption("Python · Streamlit · Pandas · Scikit-learn · Plotly")
with col_b:
    st.markdown("**📚 Course**")
    st.caption("Data Visualization & Analysis — OAMK 2026")
with col_c:
    st.markdown("**👨‍💻 Author**")
    st.caption("Edgar Ortiz — Data Engineering & AI")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; opacity: 0.5; font-size: 0.85rem;'>"
    "🌱 → 📈 → 🔗 → 🧪 → 🤖 → 🦸 &nbsp;&nbsp;·&nbsp;&nbsp; <em>Every expert was once a beginner</em>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar ──
st.sidebar.markdown("### 🗺️ Navigate the Journey")
st.sidebar.markdown(
    """
    🌱 **Week 1** — Cloud & Data Viz  
    📈 **Week 2** — Time Series  
    🔗 **Week 3** — Correlations  
    🧪 **Week 4** — Statistics  
    🤖 **Week 5** — Supervised ML  
    🦸 **Week 6** — Unsupervised ML  
    """
)
st.sidebar.markdown("---")
st.sidebar.info("Select a week above to explore the full interactive analysis.")
