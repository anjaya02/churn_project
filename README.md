# Telco Customer Churn Prediction 🚀

This project predicts customer churn for a telecom company using machine learning models. It includes full data preprocessing, model training with class imbalance handling, and a Flask-based web application for user input and live prediction.

---

## 📊 Dataset Overview

- Dataset: Telco Customer Churn
- Source: [Kaggle - Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7043 customer records and 21 features
- Target variable: `Churn` (Yes/No)

---

## 🧠 Machine Learning Workflow

1. **Exploratory Data Analysis (EDA)**
2. **Missing value handling**
3. **Feature Engineering**
   - Created `tenure_group`
   - Converted `Churn` to binary
   - One-hot encoded categorical variables
4. **Model Training**
   - Models: Decision Tree, Random Forest
   - Imbalance handled with `SMOTEENN`
   - Evaluation: Accuracy, Precision, Recall, F1-Score
5. **Best Model Selection**
   - Random Forest + SMOTEENN
6. **Model Export**
   - Saved as `model.sav`
   - Feature list saved as `features.pkl`

---

## 🌐 Flask Web App

- Accepts customer details via form
- Preprocesses data and aligns it with training features
- Predicts churn and shows result with confidence
- Pages:
  - `/` - Input form
  - `/predict` - Prediction result

---

## 🗂 Project Structure

```

churn\_project/
├── app.py                  # Flask backend
├── templates/
│   ├── index.html          # User input form
│   └── home.html           # Result page
├── model.sav               # Final saved model
├── features.pkl            # List of model input features
├── first\_telc.csv          # Dummy row used to align input features
├── tel\_churn.csv           # Final processed dataset
├── WA\_Fn-UseC\_-...csv      # Original raw dataset
├── model building.ipynb    # Notebook for model training
└── Churn Analysis.ipynb    # Notebook for EDA & preprocessing

````

---

## ⚙️ How to Run

```bash
# Install dependencies
pip install flask pandas scikit-learn imbalanced-learn

# Run the app
python app.py

# Open in browser
http://127.0.0.1:5000
````

---

## ✅ Features

* 📊 Live prediction using web form
* 🧠 Trained with imbalance-aware techniques
* 📦 Fully packaged with HTML + Flask backend
* 📁 Ready for GitHub and deployment

---

## 🧑‍💻 Author

**Anjaya Induwara**
Undergraduate | Informatics Institute of Technology 
GitHub: [@anjaya02](https://github.com/anjaya02)

---

## 📜 License

This project is licensed under the MIT License.


