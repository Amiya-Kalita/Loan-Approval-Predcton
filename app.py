from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask
app = Flask(__name__)

# Load the single pipeline model
model = joblib.load("loan_approval_model.pkl")

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # FORM INPUTS
    input_data = {
        "Gender": request.form['Gender'],
        "Married": request.form['Married'],
        "Dependents": request.form['Dependents'],
        "Education": request.form['Education'],
        "Self_Employed": request.form['Self_Employed'],
        "ApplicantIncome": float(request.form['ApplicantIncome']),
        "CoapplicantIncome": float(request.form['CoapplicantIncome']),
        "LoanAmount": float(request.form['LoanAmount']),
        "Loan_Amount_Term": float(request.form['Loan_Amount_Term']),
        "Credit_History": float(request.form['Credit_History']),
        "Property_Area": request.form['Property_Area']
    }

    df = pd.DataFrame([input_data])

    # ============================
    # MATCH EXACT TRAINING PREPROCESSING
    # ============================

    # 1 — Convert numeric
    df['Dependents'] = df['Dependents'].replace("3+", 3).astype(int)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_Income_Ratio'] = df['TotalIncome'] / df['LoanAmount']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['Balance_Income'] = df['TotalIncome'] - df['EMI']

    # 2 — One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 3 — Load the feature list used during training
    feature_names = model.feature_names_in_

    # 4 — Add missing model-trained columns (set to 0)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # 5 — Drop extra columns not used during training
    df_encoded = df_encoded[feature_names]

    # ============================
    # PREDICT
    # ============================

    pred = model.predict(df_encoded)[0]
    prob = model.predict_proba(df_encoded)[0][1]

    result = "APPROVED 😎🔥" if pred == 1 else "REJECTED ❌"

    return render_template(
        'index.html',
        prediction=result,
        probability=f"{prob:.2f}"
    )


# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
