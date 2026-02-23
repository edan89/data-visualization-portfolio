Week 3 Assignment: Correlation Analysis
Objective
Learn how to explore relationships between variables using correlation analysis.
You will calculate correlation coefficients, visualize correlations, and interpret the results in the context of your chosen dataset.
Tasks
1. Dataset Selection
Choose a dataset containing at least two numerical variables. You may use:
https://github.com/bytewax/awesome-public-real-time-datasets.git

Your dataset should contain 30+ observations to ensure meaningful correlations.
2. Exploratory Data Analysis (EDA)
Before correlation analysis, perform a quick EDA:

    Display first rows of the dataset.
    Check for missing values and decide how to handle them.
    Compute descriptive statistics for numerical variables (mean, std, min/max).
    Visualize variables:
        Histograms plots
        Scatter plots to see possible relationships

3. Correlation Computation
Determine which type of correlation test fits best your data, using normality tests
Perform correlation analysis using:

    Pearson correlation 
    Optionally Spearman or Kendall if data is non‑linear or non‑normal

Compute:

    A full correlation matrix
    Individual correlation coefficients for selected variable pairs

Highlight at least two interesting correlations:

    Strong positive
    Strong negative
    (or lack thereof, if the data behaves unexpectedly)

4. Correlation Visualization
Visualize correlations using:

    A heatmap for the correlation matrix
    Scatter plots with trend lines for important variable pairs
    Optional:
        Pairplot (Seaborn)
        Bubble plots
        Annotated heatmap

Your visualizations should help users interpret relationships clearly.
5. Interpretation and Discussion
Provide a short explanation that covers:

    What correlations are strongest?
    Are they positive or negative?
    Do they make sense given the context of the dataset?
    Could there be confounding variables?
    Are there any surprising or weak correlations?

This is the conceptual heart of the assignment.
6. Visualization (Streamlit) App (Hosted on your VPS)
Create an interactive visualization (Streamlit or other) app accessible at:

YOUR_VM_IP_ADDRESS/week3

The app must include:

    Display of basic dataset information (EDA)
    Heatmap of correlations
    Scatter plots for chosen variable pairs
    Clear explanation section

Optional enhancements:

    Dropdowns to select variables dynamically
    Toggle for Pearson vs. Spearman correlation

Submission
Submit:

    The link to your Streamlit app:
    http://YOUR_VM_IP_ADDRESS/week3
 
 These are the main points that you need to success in this assignment:
just for you to know, this is a 12 points assignment, get all the points to get a good grade.
    Grading (12 points total)
Component 	Points
Streamlit application runs correctly and is accessible 	+3
Correlation matrix + heatmap visualization 	+3
Scatter plots and meaningful analysis of correlations 	+3
General Visualization (principles below) 	+3

Students may choose any reasonable plotting style; the crucial part is the quality of communication and justification.
+1 point — Clarity and Readability
A good visualization:

    Labels axes clearly and unambiguously
    Uses readable font sizes
    Avoids clutter and unnecessary decorations
    Ensures colors or markers are distinguishable
    Has a logical layout (titles, axes, legends where appropriate)

Interpretation:
Does the viewer immediately understand what the plot shows?
+1 point — Design Choices and Best Practices
Students must demonstrate awareness of good visualization design taught in the course:

    Selecting appropriate chart types (heatmap for matrices, scatter for pairwise relationships, etc.)
    Using color intentionally, not decoratively
    Avoiding misleading encoding (e.g., incorrect scales, stretched axes)
    Ensuring that correlation strength is visually communicated (e.g., using color gradients correctly)

Interpretation:
Are the chosen visual encodings appropriate and justified?
+1 point — Interpretation and Communication
The visualization must support a meaningful explanation:

    The student discusses what the visualization reveals about the data
    Observations are connected to correlation concepts
    Prose includes insights, not just descriptions (“this is blue; this is red”)

Examples of strong communication:

    “Variables A and B show a moderately strong positive correlation (0.62), visible in the upward trend of the scatter plot.”
    “The heatmap highlights a cluster of highly correlated variables, suggesting underlying structure.”

Interpretation:
Does the visualization help tell a story about the data?
