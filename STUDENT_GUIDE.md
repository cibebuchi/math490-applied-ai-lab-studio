# MATH 490 Student Guide

Every week follows:

> **Today’s Lab → Practical Studio → Wrap-Up → My Notebook**

## Today’s Lab

Learn the week’s concept, class question, essential terms, variables, method, and expected outputs. Complete the readiness check.

## Practical Studio

Perform the instructor-led class activity.

The app loads the starting variables once. When a control is available, you may change it and the app will not force it back after each selection.

During Weeks 3 and 4, freely explore different plots and variable combinations as directed by your instructor.

Write one sentence about what you learned and mark the practical complete.

## Wrap-Up

Answer the module questions one at a time. Most weeks contain 10 questions. Week 8 and Week 16 are comprehensive review classes with 50 questions each. You will see whether each answer is correct and receive a short explanation. Your latest score and best score are recorded.

Your scheduled class ends after Wrap-Up.

## My Notebook

My Notebook is your independent assignment workspace. It does not submit work or create the final presentation for you.

### Step 1 — Plan the Assignment

Select suitable variables and write your own research question. The app does not generate the assignment question.

### Step 2 — Run My Analysis

Apply the method taught during the week and explore the controls needed to answer your question.

### Step 3 — Record Findings for Slides

Save your method, key result, interpretation, limitation, and next step. Use these notes to create the required three-slide presentation.

## Downloads

- **Readable complete notebook:** separates your independent assignment analysis from the instructor-led class analysis.
- **Continuation backup:** a JSON file used to reopen your saved work in the app.

Keep your original CSV or Excel dataset separately because the notebook backup does not contain the complete dataset file.


## Learning Library

Open **Learning Library** when you meet an unfamiliar word, model, equation, or parameter.

- Search the glossary by a word such as `probability`, `precision`, `leakage`, or `seasonality`.
- Filter by a topic group when you want to browse a part of the course.
- Open **Model Explorer** to learn how one selected model works and what each of its settings controls.
- Use **Course Map** to see how the weekly topics connect.

The library shows one selected term or model at a time so you do not have to read a long wall of definitions.

## Exam Practice

Exam Practice is private revision that you can use at any time.

1. Search for a curriculum topic.
2. Choose the topic and question types.
3. Select 5, 10, 15, or 20 questions.
4. Answer each question before seeing the correct answer.
5. Read the explanation, even when your answer is correct.
6. Review the weak concepts and study advice at the end.

The question types are:

- **Foundation:** definitions and essential ideas.
- **Application:** choosing or interpreting the correct method in context.
- **Calculation:** equations and short numerical problems.

Download the practice report when you need a record of your answers. Your topic-level scores are also included in the notebook backup and readable notebook summary.

## Full Studio

Full Studio is optional unless your instructor directs you to use it.

## Creating a visualization

1. Select the chart type.
2. Select the required variables.
3. Select **Show visualization**.
4. After changing a setting, select **Show visualization** again.

The app does not draw a default figure before you request it. On a bar chart, read the y-axis carefully: it will show a count, a mean, or a median for the numerical variable you selected.


## Reading uncertainty results

For model-based bootstrap work, train a model in the panel that appears on the Bootstrap and Uncertainty page. After the run, read each generated statement carefully:

- **Bias** compares the average bootstrap estimate with the original estimate.
- **Standard error** describes how much the estimate changes across resamples.
- **95% interval** describes uncertainty in the estimated quantity, not the spread of individual observations.
- **Cross-validation mean** describes typical performance across splits.
- **Cross-validation standard deviation** describes how stable that performance is across splits.

## Week 15 image classifiers

When you select an image classifier, read the short explanation shown above its parameters. Most options convert the image into one long row of pixel values. The **Small Convolutional Neural Network (CNN)** keeps the image as a two-dimensional color grid and learns local visual patterns such as edges, textures, and shapes. Compare the held-out accuracy, weighted F1-score, and confusion matrix rather than assuming that the most complicated model must win.

## Understanding the Week 15 evaluation audit

Read the evaluation message before comparing classifiers. A very high score is trustworthy only when the evaluation images are genuinely separate from the training images.

- **Provided validation holdout:** the ZIP's clean validation folder is preserved.
- **Final test holdout:** a separate test folder is used only for the reported final score.
- **Group-aware holdout:** related frames or exact duplicates are kept together so they cannot appear on both sides.
- **Random image-level holdout:** used only when the app cannot identify a predefined split or related groups; interpret this score more cautiously.

The app displays the number of training and evaluation images and the overlap among known groups. The expected known-group overlap is zero.

