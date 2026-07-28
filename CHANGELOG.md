# Changelog

## Version 6.16 — Learning Library and Exam Practice

- Added a searchable glossary with more than 150 key curriculum terms.
- Added topic filters and one-term-at-a-time presentation to prevent glossary overload.
- Added a Model Explorer covering 18 statistical, machine-learning, neural-network, forecasting, and computer-vision models.
- Added explanations of each model's input, output, mechanism, appropriate use, strengths, limitations, evaluation approach, and relevant parameters.
- Added a Week 1–16 course map.
- Added independent Exam Practice with searchable curriculum topics and 5–20 question sets.
- Added foundation, application, and calculation question filters.
- Added worked numerical questions for probability, descriptive statistics, regression, classification metrics, cross-validation, bootstrap, forecasting, and computer vision.
- Added one-question-at-a-time answering, immediate feedback, explanations, score advice, weak-concept summaries, and downloadable practice reports.
- Added topic-level latest and best practice scores to the project backup and readable notebook.
- Preserved the existing weekly rhythm, analytical tools, and leakage-aware image evaluation.


## Version 6.15 — Leakage-Aware Image Evaluation

- Added automatic recognition of `train`, `val`/`validation`, and `test` image folders.
- Clean predefined splits are preserved and no longer merged into a new random image split.
- Added conservative sequence-group detection for filenames such as `paper01-005.png`.
- Added exact-file SHA-256 duplicate checks across supplied splits.
- Added an evaluation audit showing complete ZIP counts, loaded counts, split contamination, known related-group coverage, training size, evaluation size, and known group overlap.
- Contaminated predefined splits are replaced with a stratified group-aware holdout when possible.
- Class-folder-only ZIPs use group-aware splitting when repeated sequence or duplicate groups are detected.
- Random image-level splitting remains available only as a clearly disclosed fallback.
- The per-class image limit is now applied separately within each source split.
- CNN early stopping uses a validation set separate from the reported evaluation set whenever needed.
- Accuracy labels now identify the real design: validation, final test, group-aware holdout, or random holdout.
- Verified the new workflow with the supplied ants-versus-bees and rock-paper-scissors ZIP files.

## Version 6.14 — Small Convolutional Neural Network for Image Classification

- Added a **Small Convolutional Neural Network (CNN)** to the classroom-trained image-classification options.
- The CNN keeps each image as a height × width × 3 grid and uses two convolution-and-pooling stages, a small dense layer, dropout, and a softmax output.
- Added early stopping and a held-out validation subset for CNN training.
- Preserved the existing logistic regression, support vector machine, random forest, k-nearest neighbors, and feedforward neural network options.
- Added a concise, model-specific teaching note that updates when the student selects an image classifier.
- Clarified which models use flattened pixel features and which model preserves spatial image structure.
- Added CNN support for classifying a newly uploaded image and displaying class probabilities.
- Added plain-language result interpretation after every classroom image-classification run.

## Version 6.13 — Pre-run Concept Teaching and Forecasting Clarity

- Added a consistent **Before you run: key terms and metrics** teaching layer across the major analytical modules.
- Expanded Week 14 forecasting definitions for horizon, chronological testing, seasonal cycles, lags, rolling windows, forecast-origin predictors, LSTM lookback, baselines, mean absolute error, and root mean squared error.
- Renamed forecasting controls to **Forecast horizon (future time steps)** and **Seasonal cycle length (rows)**.
- Added automatic detection and display of the typical time spacing between rows.
- Renamed seasonal-baseline rows to show the cycle length explicitly.
- Added a result-specific explanation identifying the lowest-error forecast and whether the selected model beats the strongest simple baseline.

## Version 6.12 — Seamless Bootstrap and Result Interpretation

- Added an embedded model-training workflow for model-metric and individual-prediction bootstrap activities.
- Removed the Week 13 dead end that only displayed “Train a model first.”
- Added regression and classification model preparation using the current dataset, target, predictors, and a held-out test set.
- Added class-probability bootstrap intervals for classification predictions.
- Added plain-language interpretation of bootstrap bias, standard error, percentile interval, interval width, and slope direction.
- Added plain-language interpretation of cross-validation mean performance, standard deviation, stability, range, and strategy.
- Extended result-reading support into Practical Studio and My Notebook.
- Preserved all Week 8 and Week 16 comprehensive review questions.

## Version 6.11 — Comprehensive Midterm and Final Review

## Added

- Expanded Week 8 to 50 unique questions reviewing Weeks 1–7.
- Added Week 16 as a 50-question final review of Weeks 9–15.
- Added rigorous artificial intelligence and machine-learning foundations, neural networks versus support vector machines, hyperparameters, tuning, regularization, overfitting, probability calculations, model evaluation, uncertainty, forecasting, computer vision, generalization, and responsible artificial intelligence.
- Made Wrap-Up question counts dynamic while preserving 10 questions for ordinary modules.
- Added dynamic retry labels and review-specific guidance.

## Preserved

- One-question-at-a-time delivery.
- Immediate correct/incorrect feedback and explanations.
- Latest score, best score, and attempt tracking.
- Existing 10-question weekly Wrap-Ups outside Weeks 8 and 16.

---

## Version 6.10 — Assignment-to-Analysis Continuity

## Saved plan carries into Run My Analysis

- Fixed the Streamlit widget-state cleanup issue that caused assignment targets and predictors to disappear between Step 1 and Step 2.
- Saved assignment variables now seed the corresponding analytical controls when students open **Run My Analysis**.
- Applied the carry-forward behavior across data questions, visualization, association, regression, classification, evaluation, cross-validation, bootstrap, and forecasting modules.
- Student changes remain free while the analysis page is active; the app does not force the saved plan back after every control change.
- Returning to the analysis step restores the saved plan when the analytical widgets have been removed from the page.

## Correct weekly method names

- Week 5 now defaults to **Simple linear regression** in the slide-evidence record.
- Week 6 now defaults to **Multiple linear regression** when multiple predictors were used.
- Method names use the actual analysis result where possible instead of exposing a generic internal estimator label.

---

## Version 6.9 — Intentional Visualizations and Clean App Identity

## Visualization workflow

- Added a blank chart-type state when no chart has been selected.
- Added a **Show visualization** button to every visualization type.
- Prevented histograms, boxplots, scatter plots, bar charts, heatmaps, and missing-data charts from appearing automatically.
- Added required-variable validation before drawing each figure.
- Preserved instructor starting variables and student-selected variables without forcing an automatic plot.

## Clear axis labels

- Replaced the generic bar-chart y-axis label `value`.
- Count charts now display **Number of observations**.
- Mean and median charts now identify the numerical variable directly, such as **Mean math score**.
- Added clearer axis labels to histograms, boxplots, scatter plots, and missing-data charts.

## Interface cleanup

- Removed the version number from the browser-tab title.
- Removed the version number from the visible footer.
- Retained the technical release number only in the project folder and documentation.

---

## Version 6.8 — Stable Controls and Student-Owned Assignments

## Stable variable selection

- Fixed the Streamlit rerun problem that forced selected variables back to the instructor’s starting values.
- Class and assignment plans now seed analytical controls only once or when the plan or dataset genuinely changes.
- Applied the fix across visualization, association, regression, classification, evaluation, cross-validation, bootstrap, and forecasting tools.
- Weeks 3 and 4 now explicitly allow free exploratory variable selection even when the instructor supplies starting variables.

## Student-owned research questions

- Removed generated research-question suggestions from My Notebook.
- Assignment research-question fields now begin blank unless the student previously saved a question.
- Students must formulate their own answerable question using the selected variables.

## Assignment and presentation workflow

- Replaced “Explain and Submit” with “Record Findings for Slides.”
- Clarified that My Notebook stores analysis evidence but does not submit work or create the final presentation.
- Made the weekly three-slide presentation assignment prominent on the My Notebook page.
- Renamed downloads as slide-preparation notes rather than finished reports.

## Clear notebook separation

- Added a readable Markdown notebook export.
- Part A contains the student’s independent assignment analysis.
- Part B contains the instructor-led class brief, practical record, class reflection, and Wrap-Up score.
- Retained a separate JSON continuation backup.

## Interface cleanup

- Removed the unnecessary “Nothing is tied to a particular variable name” message from Instructor Setup.
- Updated all visible version labels to Version 6.8.
