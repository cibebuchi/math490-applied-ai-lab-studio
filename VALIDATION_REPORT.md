# Validation Report

## Completed checks

- Python syntax compilation completed successfully.
- The full module imported successfully with a Streamlit test stub.
- The Learning Library and initial Exam Practice pages completed smoke tests with a user-interface stub.
- The completed Exam Practice result page completed a smoke test.
- The glossary contains 171 unique terms across 13 topic groups.
- The Model Explorer contains 18 model entries.
- Every exam topic contains at least 20 valid questions when all question types are selected.
- A 20-question unique attempt was generated successfully for every exam topic.
- Every practice question has a valid answer option and explanation.
- Weekly Wrap-Up counts remain unchanged:
  - 10 questions in ordinary weeks
  - 50 questions in Week 8
  - 50 questions in Week 16
- The readable notebook export now includes independent exam-practice summaries.

## Image evaluation regression tests

The two supplied image ZIP files were checked with the final parser:

### Ants versus bees

- Recognized the provided train/validation structure.
- Loaded 120 training and 120 validation images at the default 60-per-class-per-split limit.
- Used the provided train/validation split.
- Known group overlap between training and evaluation: 0.

### Rock–paper–scissors

- Detected contamination in the supplied split.
- Replaced it with a sequence- and duplicate-aware holdout.
- Produced 217 training and 83 evaluation images at the tested settings.
- Known group overlap between training and evaluation: 0.

A conventional logistic-regression image-classification smoke test completed on both datasets using reduced image limits.

## Limitations of this validation environment

- Streamlit is not installed in the execution environment, so a real browser rendering test was not possible.
- TensorFlow is not installed in the execution environment, so an end-to-end convolutional-neural-network training run was not performed here.
- The deployed host should be tested with a small image dataset before the live Week 15 class.
