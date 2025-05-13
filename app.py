import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load base dataset for reference (schema and dummies alignment)
base_df = pd.read_csv("first_telc.csv")

# Load trained model and feature list
model = pickle.load(open("model.sav", "rb"))
feature_list = pickle.load(open("features.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges']),
        'gender': request.form['gender'],
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'tenure': int(request.form['tenure'])
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Create tenure_group
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    input_df['tenure_group'] = pd.cut(
        input_df['tenure'], bins=range(1, 80, 12), right=False, labels=labels
    )
    input_df.drop(columns='tenure', inplace=True)

    # Combine with base_df to generate all dummy columns
    combined_df = pd.concat([base_df, input_df], axis=0)
    combined_df = pd.get_dummies(combined_df)

    # Align to training feature list
    final_input = combined_df.tail(1).reindex(columns=feature_list, fill_value=0)

    # Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    message = (
        "🔴 Customer is likely to churn!"
        if prediction == 1 else
        "🟢 Customer is likely to stay."
    )
    confidence = f"{probability * 100:.2f}%"

    return render_template('home.html', result=message, confidence=confidence, data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
