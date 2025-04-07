# Earnings Call Stock Ranking System (ECSRS)

## Overview

The **Earnings Call Stock Ranking System (ECSRS)** is a web-based machine learning application that predicts short-term stock movement categories (*Bullish*, *Neutral*, or *Bearish*) following a company’s earnings call. The system combines the following features:

- **Tone Sentiment Score** derived from management transcript using FinBERT.
- **Earnings Surprise**, computed from actual and estimated EPS.
- **Firm Size**, represented by market capitalization.

The ECSRS interface allows users to input earnings call data, run predictions using a trained Random Forest classifier, and explore historical prediction records through an interactive dashboard.

---

## Features

- **Interactive EPS and Firm Size Input**
- **Speaker Transcript Management (1–10 speakers)**
- **Tone Sentiment Analysis (FinBERT)**
- **Categorical Prediction using Random Forest**
- **Prediction Confidence Scores**
- **SQLite-backed History Viewer**
- **CSV Export of Historical Predictions**
- **Admin Tools (Data Deletion and Record Viewer)**

---

## How to Use

### 1. Input Financial Metrics

In the left sidebar, provide:

- **Actual EPS**: The company's reported earnings per share.
- **Estimated EPS**: Analysts' consensus EPS estimate.
- **Firm Size (Market Cap)**: Market capitalization as a numerical value.

### 2. Enter Speaker Information

In the main panel:

- Specify the number of management speakers (1–10).
- For each speaker:
  - Input **Speaker Name**.
  - Paste **Speaker Content** (transcript from the earnings call).

> **Note**: Only management speech should be included. Do not include analyst questions or external commentary.

### 3. Run Prediction

Click **"Run Prediction"** to execute the following:

- Tone analysis of management content using FinBERT.
- Earnings surprise calculation.
- Prediction of stock category using a trained Random Forest model.

The results will display:

- Predicted Category
- Weighted Tone Score
- Confidence Scores
- JSON-formatted transcript data

### 4. View and Filter Prediction History

- Scroll to the **"Past Prediction History"** section.
- Select a **date range** to filter records.
- Download the results as CSV if needed.

### 5. Explore Detailed Predictions

- Select a **Prediction ID** from the dropdown.
- View:
  - Prediction confidence scores (JSON)
  - Speaker transcripts used (JSON)

### 6. Admin Tools

- Delete all stored predictions via the **Admin Tools** expander (irreversible action).

---

## Prediction Output

- **Tone Sentiment Score**: A value between -1 and +1.
- **Predicted Category**:
  - `Bullish`: Expected short-term outperformance
  - `Neutral`: Expected moderate/uncertain behavior
  - `Bearish`: Expected underperformance
- **Confidence Scores**: Probability values for each class.

---

## Technologies Used

| Component         | Description                                |
|------------------|--------------------------------------------|
| FinBERT           | NLP model for financial sentiment analysis |
| Random Forest     | Classifier trained on semi-supervised labels|
| SQLite            | Local database for prediction logging      |
| Streamlit         | Web application framework                  |
| scikit-learn      | Machine learning model utility              |

---

## Intended Users

- Retail Investors
- Financial Analysts
- Academic Researchers
- Finance and Data Science Students

---

## Limitations

- Only text input is analyzed; voice and tone from audio is not considered.
- Results depend on input quality and coverage of speaker content.
- Market behavior is subject to external influences not captured by the model.
- The system is not intended to serve as a real-time trading signal generator.

---

## License

This project is for educational and academic use. If adapted for commercial purposes, appropriate licensing and disclaimers should be applied.

---

## Author

This system was developed as part of an undergraduate Final Year Project.

For inquiries, please contact the author or supervisor as listed in the project report.

