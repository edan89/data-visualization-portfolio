import streamlit as st

st.set_page_config(
    page_title="Data Visualization Portfolio",
    page_icon="📊",
)

st.write("# Welcome to my Data Analysis Portfolio! 📋")

st.sidebar.success("Select a week above.")

st.markdown(
    """
    ### Course Progress
    This portfolio tracks my weekly assignments for the Data Visualization course.
    
    - **Week 1:** Cloud Setup & Population Data Visualization
    - **Week 2:** Time Series Analysis (Stock Prices)
    - **Week 3:** Correlation Analysis (Multi-Stock Correlations)
    - **Week 4:** Statistical Analysis & Hypothesis Testing
    - **Week 5:** Supervised Machine Learning (Regression & Classification)
    - **Week 6:** Unsupervised Machine Learning (Clustering & Factor Analysis)
    
    Please use the sidebar to navigate to specific weeks.
    """
)
