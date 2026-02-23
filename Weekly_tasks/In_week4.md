Week 4 Assignment: Statistical Analysis and Tests
Objective
Learn how to answer concrete research questions using statistical hypothesis tests.
You will:

    Formulate hypotheses
    Choose appropriate statistical tests based on data type and assumptions
    Run tests (e.g., t‑tests, χ² tests, non‑parametric alternatives)
    Interpret p‑values, effect sizes, and confidence intervals
    Communicate results clearly with visualizations and text

1. Dataset Selection
Choose a dataset that allows you to perform at least two different statistical tests. Options:
 https://github.com/fivethirtyeight/data
 https://github.com/vincentarelbundock/Rdatasets
    
Requirements:

    At least one numerical variable and one categorical/grouping variable, OR
    Two categorical variables (for χ² tests), OR
    Two numerical variables that can be transformed into groups or used in paired designs.

The dataset should have 30+ observations to allow meaningful inference.
2. Research Questions & Hypotheses
Define 2–3 clear research questions that can be answered using statistical tests. Examples:

    Do male and female students differ in their average exam scores?
    Is there an association between smoking status and presence of a disease?
    Did performance improve after an intervention (before vs after)?

For each research question:

    Identify:
        Variables involved (type: numerical/categorical, etc.).
    Write:
        Null hypothesis (H₀)
        Alternative hypothesis (H₁)
        in clear, plain language and in statistical notation, when appropriate.

3. Assumption Checking & Test Selection
For each research question:

    Propose at least one appropriate statistical test, e.g.:
        Means / numeric outcomes
            One‑sample t‑test
            Independent samples t‑test
            Paired t‑test
            One‑way ANOVA (optionally)
            Non‑parametric alternatives: Mann–Whitney U, Wilcoxon signed‑rank, Kruskal–Wallis
        Categorical outcomes/relationships
            χ² test of independence
            χ² goodness‑of‑fit
            Fisher’s exact test (for small samples)
    Check assumptions for the chosen test(s), where relevant:
        Normality of residuals / group distributions (e.g., Shapiro–Wilk test, histograms, QQ‑plots)
        Homogeneity of variances (e.g., Levene’s test)
        Independence of observations
        Sample size (is the test reasonable?)
    If assumptions are not met, choose and justify a more robust / non‑parametric alternative.

Document your decisions:

    Which test did you choose and why?
    What did your assumption checks show?

4. Performing Statistical Tests
For each research question:

    Run the selected statistical test(s) using your preferred tools (Python, R, SPSS, etc.).
    Report at least:
        Test name (e.g., independent samples t‑test)
        Test statistic (e.g., t, F, χ², U, etc.)
        Degrees of freedom (if applicable)
        p‑value
    Effect size (required for at least one test), such as:
        Cohen’s d (for mean differences)
        r (correlation‑based effect size)
        η² / partial η² (for ANOVA)
        Cramer’s V (for χ² tests)
    Confidence intervals (CI) for at least one key result (e.g., CI for mean difference or proportion).

Clearly indicate:

    Whether H₀ is rejected or not
    But avoid making everything about “p < 0.05” only – emphasize practical significance and effect size.

5. Visualization of Test Results
Create visualizations that support and clarify your statistical tests. Examples:

    Comparing means / numeric variables
        Boxplots or violin plots by group
        Bar plots with error bars (CI or standard error)
        Dot/strip plots to show distribution within groups
    Categorical relationships
        Grouped bar charts
        Mosaic plots (optional)

Requirements:

    At least two visualizations directly linked to your hypothesis tests.
    Each visualization should:
        Have clear titles, axis labels, and legends.
        Use appropriate scales.
        Avoid unnecessary 3D effects or decorative noise.

bonus quality:

    Overlay confidence intervals
    Annotate plots with key statistics (e.g., p‑values, effect sizes)

6. Interpretation and Discussion
This is the conceptual heart of the assignment.
For each research question and test:

    Summarize the statistical result in words:
        “The independent samples t‑test showed no statistically significant difference in mean scores between Group A and Group B (t(48) = 1.12, p = 0.27, Cohen’s d = 0.20).”
    Explain:
        What does the result mean in the context of the dataset?
        Is the effect practically important, or just statistically significant?
    Reflect on:
        Are the assumptions reasonably satisfied?
        Could there be confounding variables that influence the result?
        Any limitations (sample size, measurement issues, bias)?
    Highlight:
        At least one expected result that confirms your intuition.
        At least one surprising or inconclusive result (if any).

Your explanation should go beyond describing numbers – tell a coherent story about what the tests reveal.
7. Visualization (Streamlit) App (Hosted on your VPS)
Create an interactive app (e.g., Streamlit) accessible at:

    http://YOUR_VM_IP_ADDRESS/week4

The app must include:

    Data & EDA Section
        Basic dataset info (head of the dataset, number of rows/columns)
        Summary statistics for key variables
        Basic plots (optional but recommended: histograms or bar charts for used variables)
    Statistical Tests Section
        Clear list or interface showing:
            Your research questions
            The chosen statistical test for each question
        Display of:
            Test statistic, p‑value
            Effect size (for at least one test)
            Confidence interval(s) (for at least one test)
        Short interpretation text next to the result.
    Visualization Section
        At least two plots that directly illustrate your test results (e.g., boxplot by group, bar plot of proportions).
        Short explanatory captions (what the viewer should notice).
    Explanation / Documentation Section
        Brief text explaining:
            The logic of the tests (what H₀ and H₁ are)
            Why you chose these tests
            Main conclusions

Optional enhancements:

    Dropdown menus for selecting:
        Which variables to use as groups / outcomes
        Which test to run (e.g., t‑test vs. Mann–Whitney)
    Automatic check of assumptions with small visualizations (e.g., QQ plots, histograms)
    Multi-language support (e.g., English/Finnish)

Submission
Submit in Moodle:

    The link to your Streamlit app:

        http://YOUR_VM_IP_ADDRESS/week4

Grading (12 points total)
Component 	Points
Streamlit application runs correctly and is accessible 	+3
Correct and justified use of statistical tests (incl. assumptions) 	+3
Visualizations and meaningful analysis of test results 	+3
General Visualization & Communication (principles below) 	+3
+1 point — Clarity and Readability

    Axes, titles, and legends are clearly labelled.
    Text in the app is readable (font sizes, color contrast).
    Layout is logical and easy to follow.
    The viewer can quickly understand:
        What is being tested,
        Which groups/variables are involved,
        What the main outcome is.

Interpretation question:
Does the viewer immediately understand what the test and plot are about?
+1 point — Design Choices and Best Practices
Demonstrate awareness of good statistical & visualization design (as in the course):

    Choosing the right test for the data type and design.
    Using appropriate plots:
        Boxplots/violin for distributions,
        Bar charts with CIs for proportions/means,
        Avoiding misleading scales or truncations.
    Communicating uncertainty with intervals, not just point estimates.
    Avoiding over‑interpretation of small p‑values or tiny effects.

Interpretation question:
Are the chosen tests and visual encodings appropriate and justified?
+1 point — Interpretation and Communication
Your results should be more than numbers:

    Each test result is explained in words in the context of the data.
    You discuss:
        Effect size and practical significance
        Possible confounders and limitations
    Your prose includes insights, not just descriptions.

Examples of strong communication:

    “Although the difference between groups is statistically significant (p = 0.03), the effect size is small (Cohen’s d = 0.18), suggesting limited practical relevance.”
    “The χ² test indicates a moderate association between smoking status and disease (Cramer’s V = 0.32), which is consistent with expectations from epidemiological studies.”
