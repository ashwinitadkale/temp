from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained components
model = joblib.load('pcos_model.pkl')
scaler = joblib.load('scaler.pkl')
rfecv = joblib.load('rfecv_selector.pkl')
le_dict = joblib.load('label_encoders.pkl')

# Define the feature names
feature_names = [
    'Estrogen_Level_pg_ml',
    'Testosterone_Level_ng_dl',
    'Insulin_Level_uU_ml',
    'Cycle_Length_days',
    'Cycle_Regularity',
    'Acne_Severity',
    'Hirsutism_Score',
    'Weight_Change_kg',
    'Diet_Type',
    'Physical_Activity_Level',
    'Sleep_Duration_hours',
    'Stress_Level',
    'Medication_Usage',
    'Environmental_Exposure',
    'Family_History_PCOS',
    'Ethnic_Background',
    'Data_Update_Frequency',
    'Data_Accuracy_Score',
    'Consent_Provided'
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    input_data = []
    for feature in feature_names:
        value = request.form.get(feature)
        input_data.append(value)

    # Create a DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Encode categorical variables
    for col in le_dict:
        le = le_dict[col]
        input_df[col] = le.transform(input_df[col])

    # Convert all columns to numeric
    input_df = input_df.apply(pd.to_numeric)

    # Scale the features
    input_scaled = scaler.transform(input_df)

    # Apply RFECV transformation
    input_rfe = rfecv.transform(input_scaled)

    # Make prediction
    prediction = model.predict(input_rfe)[0]
    probability = model.predict_proba(input_rfe)[0][1]

    result = 'Positive for PCOS/PCOD' if prediction == 1 else 'Negative for PCOS/PCOD'

    return render_template('result.html', prediction=result, probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
