import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def generate_sample_data():
    return pd.DataFrame({
        'income': np.random.randint(20000, 100000, 500),
        'credit_score': np.random.randint(300, 850, 500),
        'academic_performance': np.random.randint(50, 100, 500),
        'loan_approved': np.random.choice([0, 1], 500)  # 1 = Approved, 0 = Denied
    })

# Generate dataset
data = generate_sample_data()
X = data[['income', 'credit_score', 'academic_performance']]
y = data['loan_approved']

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, 'loan_approval_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')

def check_loan(income, credit_score, academic_performance):
    input_data = np.array([[income, credit_score, academic_performance]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    reason = "Key Factors Influencing Decision:<br>"
    feature_names = ['Income', 'Credit Score', 'Academic Performance']
    importances = model.feature_importances_
    for i, val in enumerate(importances):
        reason += f"{feature_names[i]}: {val:.2f}<br>"
    
    if prediction == 1:
        return f"Loan Approved!<br>Approval Confidence: {proba[1]:.2%}<br><br>{reason}"
    else:
        return f"Loan Denied.<br>Rejection Confidence: {proba[0]:.2%}<br><br>{reason}"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        try:
            income = float(request.form['income'])
            credit_score = float(request.form['credit_score'])
            academic_performance = float(request.form['academic_performance'])
            result = check_loan(income, credit_score, academic_performance)
        except ValueError:
            result = "Invalid input! Please enter numerical values."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
