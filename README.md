# Telco Customer Churn Prediction 🚀

This project predicts customer churn for a telecom company using machine learning models. It includes full data preprocessing, model training with class imbalance handling, and a Flask-based web application for user input and live prediction.

---

## 📊 Dataset Overview

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Source file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7043 customer records and 21 features
- Target variable: `Churn` (Yes/No)

---

## 🧠 Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Class imbalance check
   - Insight extraction (e.g., impact of monthly charges on churn)

2. **Data Cleaning**
   - Converted `TotalCharges` to numeric
   - Removed missing rows (~0.15%)

3. **Feature Engineering**
   - Created `tenure_group` from `tenure`
   - One-hot encoded categorical variables

4. **Model Training**
   - Used `DecisionTreeClassifier` and `RandomForestClassifier`
   - Applied **SMOTEENN** to handle imbalance
   - Achieved ~94% accuracy on balanced test data

5. **Model Export**
   - Final model saved as `model.sav`
   - Trained feature list saved as `features.pkl`

---

## 🌐 Web Application (Flask)

- Built using Flask and HTML (with Bootstrap styling)
- `index.html`: input form for user details
- `home.html`: shows prediction result and confidence
- Live inference using `model.sav` and aligned features

---

## 🛠 Folder Structure

churn_project/
├── app.py # Flask backend
├── templates/
│ ├── index.html # Input form
│ └── home.html # Prediction result page
├── tel_churn.csv # Preprocessed dataset
├── model.sav # Trained model
├── features.pkl # List of input features
├── model building.ipynb # Full model training notebook
├── Churn Analysis.ipynb # EDA and preprocessing notebook
├── first_telc.csv # Template used for dummy alignment
└── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original dataset
