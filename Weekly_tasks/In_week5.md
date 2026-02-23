Week 5 Assignment: Supervised Machine Learning – Regression & Classification
Objective
Learn how to build, evaluate, and interpret supervised machine learning models for:

    Regression (predicting a numerical value)
    Classification (predicting a categorical label)

You will:

    Define prediction tasks (one regression, one classification)
    Prepare data for modeling (train–test split, basic preprocessing)
    Train and compare at least two models per task
    Evaluate models using appropriate metrics
    Visualize and interpret the results
    Deploy an interactive Streamlit app on your VPS

You may continue with the same dataset from previous weeks if it fits the requirements, or choose a new one.
1. Dataset Selection
Choose a dataset that allows you to perform both:
    At least one regression task (numeric target variable)
    At least one classification task (categorical target variable)
Possible sources:
 https://github.com/bytewax/awesome-public-real-time-datasets.git
 https://github.com/fivethirtyeight/data
 https://github.com/vincentarelbundock/Rdatasets

Requirements
Your dataset should:
    Contain at least 200 observations (rows)
    Include multiple predictor variables (features)
    Have:
        At least one numeric variable that can be used as a regression target
        At least one categorical variable (with 2–10 classes) that can be used as a classification target

If needed, you may:
    Create a categorical variable by binning a numeric one (e.g., “low/medium/high income”)
    Filter the data to focus on a meaningful subset

2. Prediction Tasks & Questions
Define two prediction tasks:

    Regression Task
        Example:
            “Can we predict a student’s final exam score based on their study time, attendance, and previous grades?”
            “Can we predict house prices based on size, location, and number of rooms?”
    Classification Task
        Example:
            “Can we classify whether a customer will churn based on their usage statistics?”
            “Can we classify whether a passenger survived (yes/no) based on demographic and ticket features?”

For each task, clearly specify:

    Target variable (what you want to predict; type: numeric or categorical)
    Input features (which variables you use as predictors and why)
    Short problem description in plain language, including:
        Why this prediction would be useful
        What kind of improvement over a trivial/baseline model you might expect

Write this as a short “Problem definition” section for each task.
3. Data Preparation & Splitting
For each task (regression and classification):

    Data Cleaning / Preprocessing
        Handle missing values (e.g., drop, impute with mean/median/mode)
        Convert categorical variables to numeric form (e.g., one-hot encoding)
        Optionally:
            Scale/standardize numeric features (useful for k-NN, logistic regression, etc.)
            Remove clearly irrelevant or highly collinear features (if justified)
    Train–Test Split
        Split your data into training and test sets (e.g., 70% train, 30% test)
        Ensure the split is random and that the test set is not used for model training or hyperparameter tuning
    (Optional but recommended) Cross-Validation
        For at least one model, use k-fold cross-validation on the training set
        Report the mean and standard deviation of the metric across folds

Document your choices:

    How did you handle missing values?
    Did you standardize/scale features? Why or why not?
    What train–test split ratio did you use?

4. Regression Modeling
For your regression task:
4.1 Baseline Model
Create a simple baseline model, such as:

    Always predicting the mean of the target (or median)
    Or a very simple linear model with only one feature

Compute at least one metric for this baseline (e.g., MAE, MSE, RMSE, or R²).
4.2 At Least Two Regression Models
Train and compare at least two different regression models, for example:

    Linear Regression
    Ridge or Lasso Regression
    k-Nearest Neighbors Regressor
    Decision Tree Regressor
    Random Forest Regressor
    Gradient Boosting / XGBoost / LightGBM (if comfortable)

For each model, report:

    Model type and key hyperparameters (e.g., tree depth, k for k-NN)
    Evaluation on the test set using at least two metrics, e.g.:
        RMSE (Root Mean Squared Error)
        MAE (Mean Absolute Error)
        R² (Coefficient of Determination)

4.3 Model Comparison & Interpretation

    Compare models against each other and the baseline
    Highlight:
        Which model performed best and by how much
        Whether improvements are practically meaningful
    Optionally:
        Inspect feature importance (for tree-based models)
        Plot predicted vs. actual values

5. Classification Modeling
For your classification task:
5.1 Baseline Model
Create a baseline classifier, for example:

    Always predicting the majority class
    Or a very simple model with a single feature

Report baseline metrics, such as:

    Accuracy
    Class distribution (to show if the data is imbalanced)

5.2 At Least Two Classification Models
Train and compare at least two different classification models, for example:

    Logistic Regression
    k-Nearest Neighbors Classifier
    Decision Tree Classifier
    Random Forest Classifier
    Support Vector Machine (SVM)
    Gradient Boosting Classifier / XGBoost

For each model, evaluate on the test set using at least three of:

    Accuracy
    Precision
    Recall
    F1-score
    Confusion matrix
    ROC curve and AUC (for binary classification)

If the classes are imbalanced, discuss:

    Why accuracy alone can be misleading
    Which metrics are more informative (e.g., F1, recall)

5.3 Model Comparison & Interpretation

    Compare your models to each other and to the baseline
    Highlight:
        Which model is most accurate
        Which model is best in terms of F1 / recall / precision (depending on the problem)
    Provide interpretation in the context of the problem:
        E.g., “Missing some positive cases is more serious than false alarms, so recall is more important.”

6. Visualization of Models & Results
Create visualizations that directly support your modeling results.
Regression Visualizations (at least one)
Examples:

    Predicted vs. actual scatter plot with a 45-degree line
    Residual plot (errors vs. predicted or vs. a key feature)
    Bar chart of feature importances (for tree-based models)

Classification Visualizations (at least one)
Examples:

    Confusion matrix heatmap
    ROC curves for multiple models on the same plot
    Bar chart of feature importances (for tree-based models)

Requirements:

    At least two visualizations total, covering both regression and classification
    Each plot should have:
        Clear titles, axis labels, and legends
        Appropriate scales
        Short captions explaining what the viewer should notice

(Extra quality: annotate plots with metrics, e.g., “Test RMSE = 4.2” or “AUC = 0.87”.)
7. Interpretation and Discussion
This is the conceptual heart of the assignment.
For each task (regression & classification):

    Summarize the best model in words, e.g.:

        “The random forest regressor achieved the lowest error (RMSE = 4.2, MAE = 3.1), clearly outperforming both the baseline and the linear regression model.”

        “The logistic regression classifier outperformed the baseline and k-NN in terms of F1-score (F1 = 0.78 vs. 0.65 and 0.70), suggesting it is the most balanced model for this task.”

    Interpretation in context:
        What do these results mean for the real-world problem?
        Are the improvements over the baseline practically important?
        Which errors are most serious (false positives vs. false negatives)?
    Reflection:
        Any overfitting signs? (e.g., train vs. test performance)
        Limitations of the data (size, bias, missing features)
        Potential confounding variables or unmeasured factors
        What you would try next (more features, different models, tuning, regularization, etc.)

Highlight:

    At least one result that matches your expectations
    At least one surprising or inconclusive result

8. Visualization (Streamlit) App (Hosted on your VPS)
Create an interactive app (e.g., Streamlit) accessible at:

    http://YOUR_VM_IP_ADDRESS/week5

The app must include:
A. Data & EDA Section

    Basic dataset info:
        Head of the dataset
        Number of rows/columns
    Summary statistics for key variables
    At least one simple plot (e.g., histogram or bar chart for a key feature)

B. Modeling Section
Clearly separate Regression and Classification:
For each:

    Show:
        The prediction task description
        List of models you trained (including baseline)
    For the selected model (or all models in a table), display:
        Evaluation metrics (e.g., RMSE, R² for regression; accuracy, F1, etc. for classification)
        Short text interpretation next to the results

Optional (recommended):

    Dropdowns to:
        Choose the target variable (if you defined multiple)
        Select which features to include
        Select which model to run

C. Visualization Section

    At least one plot for the regression results (e.g., predicted vs. actual)
    At least one plot for the classification results (e.g., confusion matrix or ROC curve)
    Short captions explaining:
        What the viewer is seeing
        What the plot implies about model performance

D. Explanation / Documentation Section
Brief, readable text explaining:

    What supervised learning is (briefly)
    What the regression and classification tasks are in your dataset
    Which models were used and why
    Key conclusions:
        Which models perform best
        Whether the performance is practically useful

Optional enhancements:

    Automatically computed train vs. test performance to illustrate overfitting
    Sliders for adjusting hyperparameters (e.g., tree depth, k in k-NN)
    Multi-language support (e.g., English/Finnish)

Submission
Submit in Moodle:

    The link to your Streamlit app:

        http://YOUR_VM_IP_ADDRESS/week5

Grading (12 points total)
Component 	Points
Streamlit app runs correctly and is accessible 	+3
Correct and justified use of ML models & evaluation metrics 	+3
Visualizations and meaningful analysis of model results 	+3
General Visualization & Communication 	+3
+1 point — Clarity and Readability

    Titles, axes, and legends are clearly labelled
    Text is readable (font size, color contrast)
    Layout is logical and easy to follow
    The viewer can quickly understand:
        What is being predicted
        Which features and models are used
        What the main conclusions are

+1 point — Modeling Choices and Best Practices

    Models match the type of target (regression vs classification)
    Baseline models are included and used for comparison
    Appropriate metrics are used:
        Error metrics and R² for regression
        Classification metrics beyond accuracy when needed
    Awareness of:
        Overfitting (train vs. test performance)
        Class imbalance (if present)

+1 point — Interpretation and Communication

    Results are explained in words in the context of the dataset
    You discuss:
        Practical significance (not only numeric scores)
        Strengths and weaknesses of different models
        Limitations and possible improvements
    Your prose includes insights, not just metric listings, e.g.:

        “Although the random forest achieved slightly better RMSE than linear regression, the gain is small (RMSE improvement ≈ 5%), and the added complexity may not be justified.”

        “Because false negatives are more costly in this application, we prefer the model with higher recall, even though its overall accuracy is lower.”

