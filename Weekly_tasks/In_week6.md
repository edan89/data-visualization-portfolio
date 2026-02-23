Week 6 Assignment: Unsupervised Machine Learning – Clustering & Factor Analysis
Objective
This week you will learn how to apply unsupervised machine learning techniques to explore hidden structure in data.
You will:

    Perform dimensionality reduction using Factor Analysis (and optionally PCA for comparison)
    Perform clustering using at least two methods (e.g., k‑means, hierarchical, DBSCAN)
    Interpret and visualize latent structures and cluster patterns
    Evaluate clustering quality using metrics (e.g., silhouette score)
    Build a Streamlit app on your VPS for interactive exploration of factors and clusters

You may continue using the same dataset from previous weeks (recommended), provided it contains several numeric variables. Otherwise, choose a new one.
1. Dataset Selection
Choose a dataset with multiple numeric variables suitable for:

    Factor Analysis (latent variables)
    Clustering (grouping observations)

Requirements
Your dataset must contain:

    At least 200 observations
    At least 6–8 numeric variables
    Reasonable correlations between some variables (Factor Analysis requires covariance)

Possible data sources
 Use some data that you haven't used in the previous weeks assignments.
 Possible sources:
 https://github.com/bytewax/awesome-public-real-time-datasets.git
 https://github.com/fivethirtyeight/data
 https://github.com/vincentarelbundock/Rdatasets

Notes

   
    If your dataset contains categorical variables, you may drop or encode them but unsupervised tasks can only use numeric features.
    If needed, scale/normalize variables.

2. Unsupervised Learning Tasks & Questions
This assignment has two tasks:
Task A: Factor Analysis (FA)
Define a question such as:

    “Can we reduce 15 student performance metrics into a smaller number of latent factors?”
    “Is it possible to uncover underlying behavioral dimensions in customer activity data?”
    “Can product rating attributes be summarized into a few meaningful latent constructs?”

Specify:

    Which variables you include and why
    The goal: compression? interpretation? noise reduction?
    What insights FA might reveal

Write a short “Problem definition” similar to Week 5.
Task B: Clustering
Define a clustering problem such as:

    “Can customers be grouped into distinct segments based on their behavior?”
    “Are there meaningful groups of students with similar learning profiles?”
    “Can we find natural clusters in health or sensor data?”

Specify:

    Variables used
    Why clustering might be useful (personalization, subgroup discovery, anomaly detection)
    What you might expect (e.g., 3–5 clusters, separation along dominant features)

3. Data Preparation & Preprocessing
For both FA and clustering:
Data Cleaning

    Handle missing values (drop rows/columns or impute)
    Convert categorical variables if needed
    Remove clearly irrelevant or low‑variance features
    Optionally remove highly correlated variables for clustering (not required)

Standardization
Required: Standardize numeric features (mean=0, variance=1)
Reason: FA and clustering are scale‑sensitive.
Train–Test Splits?
Not needed (unsupervised).
But you must:

    Fit models on full dataset
    If you visualize clusters on PCA/FA space, apply transformations correctly

Document your preprocessing choices.
4. Factor Analysis (Task A)
4.1 Determine Number of Factors
Use at least one method:

    Kaiser criterion (eigenvalues > 1)
    Scree plot
    Parallel analysis (optional, advanced)

Visualize eigenvalues / explained variance.
4.2 Fit the Factor Analysis Model
Perform FA using your selected number of factors.
Report:

    Factor loadings matrix
    Communalities
    Variance explained

Interpret each factor in plain language. Example:

    “Factor 1 loads strongly on math_score, physics_score, and study_time → Academic Performance Factor.”

4.3 Optional: Compare with PCA
(Not required but recommended.)
Compare:

    Interpretability
    Variance explained
    Whether factors make more sense than principal components

4.4 Visualize Factors
At least one plot:

    Factor loading plot / heatmap
    Factor score scatter plot
    Correlation circle (if PCA is included)

Explain what the viewer should notice.
5. Clustering (Task B)
5.1 Choose at Least Two Clustering Algorithms
Examples:

    k-means
    Hierarchical clustering (Ward / complete / average)
    DBSCAN
    Gaussian Mixture Models (GMM)

For each method:

    Describe key hyperparameters (e.g., k for k-means)
    Fit clusters on standardized data

5.2 Determine Number of Clusters
Use at least one:

    Elbow method (plot required)
    Silhouette analysis
    Dendrogram cut for hierarchical clustering

5.3 Evaluate Clusters
Use metrics such as:

    Silhouette score
    Calinski–Harabasz index
    Davies–Bouldin index

Compare models using 2–3 metrics.
5.4 Cluster Interpretation
Explain:

    What distinguishes clusters?
    Which variables have the strongest influence?
    Are clusters well separated?
    Are clusters meaningful or just statistical artifacts?

5.5 Cluster Visualizations
At least one required, such as:

    Scatter plot of clusters in FA/PCA space
    Dendrogram for hierarchical clustering
    Heatmap of mean feature values per cluster

Include captions and explain the insights.
6. Joint Interpretation of Factors & Clusters
A short section where you connect both tasks:

    Do certain clusters correspond to certain factors?
    Are clusters separated along Factor 1 / Factor 2?
    Do latent factors provide clearer separation?

Example:

    “Clusters were mainly separated along the ‘academic performance’ factor, suggesting this is the dominant underlying dimension in the dataset.”

7. Discussion & Reflection
Discuss:

    Were clusters meaningful?
    Did FA reveal intuitive latent factors?
    Were results stable across methods?
    Potential overfitting signs (e.g., too many factors)
    Limitations (data quality, sample size, noise)
    Next steps (tuning clustering, adding variables, rotating factors)

Highlight:

    One expected result
    One surprising result

8. Visualization (Streamlit) App (Hosted on VPS)
Available at:
http://YOUR_VM_IP_ADDRESS/week6
The app must contain:
A. Data & EDA Section
Show:

    Dataset preview
    Summary statistics
    Correlation matrix
    At least one plot (histogram, heatmap, etc.)

B. Factor Analysis Section
Include:

    Number of factors selected
    Factor loadings (table)
    Factor scores
    At least one FA visualization
    Short interpretation text

Optional:

    Dropdown to choose number of factors
    PCA comparison toggle

C. Clustering Section
Include:

    Choice of clustering algorithm
    Choice of k (if applicable)
    Evaluation metrics
    Cluster scatterplot (FA/PCA space)
    Dendrogram (if hierarchical)

Add short commentary next to results.
D. Documentation / Learning Section
Plain-language explanation:

    What is unsupervised learning?
    What are factors and clusters?
    Why standardization matters?
    Key findings and conclusions

Optional extras:

    Hyperparameter sliders
    Multi-language support (FI/EN)
    Ability to download factor scores or cluster assignments

Submission
Submit in Moodle:

    Link to Streamlit app

Grading Rubric (12 points total)
Component 	Points
Streamlit app runs correctly 	+3
Correct use of unsupervised models & justified parameter choices 	+3
Visualizations & meaningful analysis 	+3
Quality of documentation & communication 	+3