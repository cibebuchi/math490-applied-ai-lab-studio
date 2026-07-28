# Deployment Guide

## Recommended deployment: Streamlit Community Cloud

1. Create or open a GitHub repository.
2. Upload the project files so `app.py`, `requirements.txt`, and the `.streamlit` folder are at the repository root.
3. Commit and push the files.
4. Sign in to Streamlit Community Cloud with the GitHub account that can access the repository.
5. Create a new app and select:
   - Repository: the MATH 490 repository
   - Branch: `main`
   - Main file path: `app.py`
6. Deploy and wait for the dependency installation to complete.
7. Open the app and test Weeks 1, 8, 14, 15, and 16, Learning Library, and Exam Practice.

## Resource considerations

- The app allows uploads up to 500 MB through `.streamlit/config.toml`.
- TensorFlow and SHAP make the dependency installation and app startup heavier than a basic Streamlit app.
- Use modest image sizes and per-class limits during live classes.
- The small convolutional neural network should be tested on the deployed host before class.
- A Streamlit session is not a permanent database. Students should download their notebook backup and reports.

## Canvas workflow

The simplest initial integration is to place the deployed app link in the relevant Canvas modules. Students complete the practical and notebook work in the app, then submit the required slides or requested evidence through Canvas.

The app does not currently use Learning Tools Interoperability, automatic Canvas login, or automatic grade transfer.

## Deployment checklist

- App opens without a dependency error.
- Demo datasets load.
- CSV and Excel uploads work.
- Learning Library search works.
- Exam Practice starts, scores, and downloads a report.
- Week 15 image ZIP audit identifies the evaluation design.
- Convolutional neural-network training works on a small dataset.
- Notebook JSON backup can be downloaded and imported.
