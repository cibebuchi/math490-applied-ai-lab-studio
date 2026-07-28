# MATH 490 Applied AI Lab Studio — Version 6.16

A Streamlit teaching environment for **Introduction to the Mathematics of Data Science, Artificial Intelligence, and Machine Learning**.

Every week follows the same rhythm:

> **Instructor Setup → Today’s Lab → Practical Studio → Wrap-Up → My Notebook**

The scheduled class ends at **Wrap-Up**. **My Notebook** is the student’s independent assignment workspace after class.

## What Version 6.16 adds

### Searchable Learning Library

- Added a clean **Learning Library** outside the weekly workflow.
- The glossary contains more than 150 curriculum terms across artificial intelligence, probability, data, visualization, regression, classification, validation, uncertainty, neural networks, forecasting, computer vision, and responsible artificial intelligence.
- Students search by word or topic group, choose one matching term, and see only that definition, why it matters, notation, and related terms.
- Added a **Model Explorer** for the models used in the app, including linear and regularized regression, logistic regression, discriminant analysis, trees, ensembles, nearest neighbors, support vector methods, feedforward neural networks, long short-term memory networks, convolutional neural networks, and pretrained MobileNetV2.
- Each model page explains the input, output, mechanism, appropriate use, strengths, limitations, evaluation approach, and every relevant teaching parameter.
- Added a week-by-week course map so students can see how the curriculum progresses.

### Independent Exam Practice

- Added a separate **Exam Practice** space that does not require completion of a practical activity.
- Students can search the curriculum, choose a topic, select foundation, application, and calculation questions, and answer 5, 10, 15, or 20 questions.
- Questions appear one at a time with immediate feedback and explanations.
- The topic pools reuse the reviewed weekly question banks and add worked numerical questions for probability, descriptive statistics, regression, classification metrics, validation, bootstrap, forecasting, and computer vision.
- After each attempt, the app reports the score, percentage, missed concepts, targeted study advice, and a downloadable answer report.
- Topic-level latest and best scores are preserved in the notebook backup and summarized in the readable notebook.

### Deployment readiness

- Added a deployment guide for GitHub and Streamlit Community Cloud.
- The visible app remains free of version numbers; release numbers remain only in project files for maintenance.
- All Version 6.15 leakage-aware image evaluation behavior is preserved.

## What Version 6.15 adds

### Leakage-aware image evaluation

- Week 15 now audits image ZIP structure before training any classifier.
- Clean `train/class`, `val/class`, and optional `test/class` folders are preserved rather than merged.
- Related filename sequences and exact duplicate image bytes are checked across supplied splits.
- When a supplied split is contaminated, the app rebuilds a sequence- and duplicate-aware holdout that keeps known related groups entirely on one side.
- Class-folder-only ZIPs use group-aware splitting when repeated groups are detected and a clearly disclosed stratified random image holdout only as the fallback.
- Reported metrics are labeled **Validation accuracy**, **Final test accuracy**, **Group-aware holdout accuracy**, or **Random holdout accuracy** according to the actual evaluation design.
- The image limit is applied separately to every class within every source split, preventing the training folder from consuming the validation quota.

### Pre-run concept teaching across the app

- Major analytical activities now begin with an expanded **Before you run: key terms and metrics** section.
- Students see the meaning of relevant controls, outputs, metrics, uncertainty terms, baselines, and common evaluation ideas before running an analysis.
- The teaching layer covers probability, visualization, association, regression, classification, predictor selection, model explanations, model comparison, cross-validation, bootstrap, forecasting, and computer vision.
- The classroom computer-vision lab now includes a **Small Convolutional Neural Network (CNN)** alongside logistic regression, support vector machines, random forest, k-nearest neighbors, and a feedforward neural network.
- A model-specific note explains how the currently selected image classifier reads the image, makes a decision, and differs from the other approaches.

### Clearer Week 14 forecasting

- **Forecast horizon** is now labeled as future time steps.
- **Seasonal cycle length** is explicitly measured in rows.
- The app detects and explains the typical time spacing between rows.
- Naïve latest-value, seasonal naïve, and expanding historical mean baselines are defined before the run.
- The post-run explanation identifies the lowest-error method and states whether the selected model beat the strongest simple baseline.

## Previous Version 6.12 improvements

### Comprehensive review modules

- **Week 8** now contains **50 unique midterm-review questions** covering Weeks 1–7, including artificial intelligence, machine learning, neural networks, support vector machines, tuning, regularization, probability calculations, data questions, visualization, association, regression, and machine-learning regression.
- **Week 16** is a new final-review module with **50 unique questions** covering Weeks 9–15, including classification, thresholds, confusion-matrix metrics, model comparison, neural networks, cross-validation, bootstrap uncertainty, forecasting, long short-term memory networks, computer vision, generalization, and responsible artificial intelligence.
- All other weekly modules retain **10 questions**.
- Review questions still appear one at a time, provide immediate feedback and explanations, and record the latest score, best score, and attempt count.


### Self-contained Week 13 bootstrap workflow

- Model-based bootstrap choices now include an embedded model-training step.
- Students can train or replace a regression or classification model without leaving the Bootstrap and Uncertainty page.
- Saved model metrics and individual predictions can then be bootstrapped directly.
- Classification prediction uncertainty is expressed as a selected class probability rather than an encoded class number.

### Result-reading support

- Cross-validation now explains the mean score, standard deviation, stability, score range, and validation strategy.
- Bootstrap now explains the original estimate, estimated bias, bootstrap standard error, percentile interval, and interval width.
- These statements also flow into Practical Studio and My Notebook so students can use the evidence when preparing their slides.

## Previous Version 6.11 improvements

### Assignment plans now carry into the analysis

Across the weekly modules, the target, predictor or predictors, and other relevant choices saved in **Plan the Assignment** now appear as the starting values in **Run My Analysis**. Students may still change those values after the analysis opens. Returning to the analysis step restores the saved plan instead of unrelated generic defaults.

### Week-specific method names

The findings step now uses the method that matches the weekly activity. In particular:

- Week 5 displays **Simple linear regression**.
- Week 6 displays **Multiple linear regression** when multiple predictors were analyzed.

The method label is based on the actual saved analysis where possible rather than the generic internal model name.


### Visualizations now wait for the student

The visualization page no longer draws a chart automatically. Students first choose the chart type and required variables, then select **Show visualization**. Changing a setting requires selecting the button again, so the app never presents an unintended default figure.

Bar-chart axes are now explicit:

- Count bars use **Number of observations** on the y-axis.
- Mean bars use labels such as **Mean math score**.
- Median bars use labels such as **Median math score**.

### Cleaner product identity

The browser-tab title and app footer no longer display a version number. The technical version remains only in the downloadable project folder and documentation.

### Variable choices no longer jump back

The instructor’s plan now loads starting variables only once. Streamlit reruns no longer force the chart, x-variable, y-variable, target, predictors, or model controls back to their original values after a student changes them.

Weeks 3 and 4 are intentionally exploratory. The instructor may provide starting variables, but students can freely change the variables, plot type, grouping, and association controls during the practical.

### Students write their own assignment research questions

My Notebook no longer generates or displays a suggested assignment question. Students select their variables and write one clear, answerable research question themselves.

Instructor-generated questions remain available in Instructor Setup for preparing the class lesson.

### My Notebook supports slide preparation rather than submission

The weekly assignment is displayed prominently as a **three-slide presentation assignment** tied to the module. My Notebook now follows:

1. **Plan the Assignment**
2. **Run My Analysis**
3. **Record Findings for Slides**

The app saves the student’s research question, data choices, method, key result, interpretation, limitation, and next step. Students use those notes to create and present the final slides outside the app.

### Downloaded notebook sections are clearly separated

The readable notebook download contains:

- **Part A — Student Independent Assignment Analysis**
- **Part B — Instructor-Led Class Analysis and Practical Record**
- **Part C — Independent Exam Practice Summary**

A separate JSON backup remains available so students can continue their work in a later session.

## Weekly learning journey

### Instructor Setup — Prepare

The instructor selects the class dataset, research question, variables, student choice level, practical task, required outputs, and class duration.

### Today’s Lab — Learn

Students learn the weekly concept, research purpose, essential terms, expected outputs, and complete a short readiness question.

### Practical Studio — Practise

Students perform the instructor-led analysis. The class plan supplies the starting point, but unlocked controls remain fully interactive.

### Wrap-Up — Reflect

Students answer module-specific questions one at a time. Weeks 1–7 and 9–15 contain 10 questions. Week 8 and Week 16 contain 50-question comprehensive reviews. Each answer receives immediate feedback and a short explanation. The app records the latest score, best score, and attempt count.

### My Notebook — Apply independently

Students use the assigned dataset or activity to formulate their own research question, perform the weekly analysis, record evidence, and prepare their presentation.

### Learning Library — Review concepts

Students search course terms, inspect one model and its parameters at a time, and use the course map without changing any dataset.

### Exam Practice — Prepare independently

Students complete searchable topic-based practice sets of 5–20 questions and receive score-based study advice.

## Week 1 and probability

Week 1 uses a coin-toss simulator to introduce:

- Experiments and outcomes
- Sample space
- Theoretical probability
- Experimental probability
- Random fluctuation
- The effect of increasing the number of trials

Week 2 then begins the main semester project with research questions, targets, predictors, and leakage.

## Review weeks

### Week 8 — Midterm Preparation Lab

The 50-question review integrates Weeks 1–7 and includes conceptual and numerical questions.

### Week 16 — Final Review

The 50-question final review integrates classification, evaluation, validation, uncertainty, forecasting, neural networks, and computer vision from Weeks 9–15.

## Dataset-responsive design

The app does not require any fixed variable name. When a dataset changes, it reads the new columns and data types, rebuilds compatible choices, clears stale results, and warns when a saved plan no longer matches the data.

The class, assignment, and Full Studio datasets remain separate.

## Optional Full Studio

Full Studio retains the complete analytical collection, including probability, visualization, association, regression, classification, model comparison, cross-validation, bootstrap uncertainty, forecasting, neural networks, computer vision, permutation importance, and SHAP.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

See `DEPLOYMENT.md` for the GitHub and Streamlit Community Cloud checklist.

## Canvas use

Deploy the app independently and link it from Canvas. Canvas handles announcements, due dates, grades, slide submission, and presentation instructions. The app handles teaching, practical analysis, Wrap-Up checks, and the student’s analysis notebook.
