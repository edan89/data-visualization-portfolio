# 📊 Data Visualization Portfolio

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A 6-week interactive data analysis portfolio built with **Streamlit**, covering everything from basic visualization to machine learning — designed to showcase practical data skills.

---

## 🗂️ Weekly Breakdown

| Week | Topic | Key Skills |
|------|-------|------------|
| **1** | Cloud Setup & Population Data | Streamlit basics, line charts, bar plots |
| **2** | Time Series Analysis | Stock prices with yfinance, interactive Plotly charts |
| **3** | Correlation Analysis | Multi-stock correlations, heatmaps, scatter plots |
| **4** | Statistical Analysis | Hypothesis testing (t-tests, χ²), effect sizes, confidence intervals |
| **5** | Supervised ML | Regression & Classification (Linear, Logistic, Random Forest) |
| **6** | Unsupervised ML | Clustering (K-Means, DBSCAN) & Factor Analysis |

---

## 🛠️ Tech Stack

- **Framework:** Streamlit (multipage app)
- **Data:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Statistics:** SciPy, Statsmodels
- **Finance Data:** yfinance

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/data-visualization-portfolio.git
cd data-visualization-portfolio

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run Home.py
```

The app opens at `http://localhost:8501`. Use the sidebar to navigate between weeks.

---

## 📁 Project Structure

```
├── Home.py                 # Main entry point
├── pages/
│   ├── week1.py            # Population visualization
│   ├── week2.py            # Stock time series
│   ├── week3.py            # Correlation analysis
│   ├── week4.py            # Statistical testing
│   ├── week5.py            # Supervised ML
│   └── week6.py            # Unsupervised ML
├── data/
│   └── population_country_columns.csv
├── tutorials/              # Learning notes & tutorials
├── Weekly_tasks/            # Assignment descriptions
├── requirements.txt
└── README.md
```

---

## 📄 License

This project is for educational purposes as part of the **Data Visualization** course at OAMK (Oulu University of Applied Sciences).

---

## 👤 Author

Built with ❤️ as part of my **Master's in Data Analytics & Project Management** studies.
