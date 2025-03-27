import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Load your trained models and scaler.
dt_model = pickle.load(open("dt_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__, static_url_path="/static")

def get_expected_features():
    # Use the feature names that were used when training the scaler/model.
    if hasattr(scaler, "feature_names_in_"):
        return scaler.feature_names_in_.tolist()
    return ["income", "credit_score", "gpa"]  # Adjust as per your original training

# Route to serve the main HTML page using Flask templates
@app.route("/")
def index():
    return render_template("index.html")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        income = float(data.get("income", 0))
        credit_score = int(data.get("credit_score", 0))
        gpa = float(data.get("gpa", 0))
    except Exception:
        return jsonify({"error": "Invalid input data."}), 400

    expected_features = get_expected_features()
    input_dict = {key: value for key, value in {
        "income": income,
        "credit_score": credit_score,
        "gpa": gpa,
        # Extra features are ignored if not expected.
        "debt_to_income_ratio": 0.2,
        "household_income": income + 10000,
        "part_time_income": 3000,
        "financial_support": 10000,
        "student_debt": 15000,
        "tuition_cost": 25000
    }.items() if key in expected_features}
    
    input_data = pd.DataFrame([input_dict])
    X_scaled = scaler.transform(input_data)
    dt_pred = int(dt_model.predict(X_scaled)[0])
    lr_pred = int(lr_model.predict(X_scaled)[0])

    return jsonify({
        "decision_tree_prediction": dt_pred,
        "logistic_regression_prediction": lr_pred
    })

# Visualization endpoint
@app.route("/generate_visualizations", methods=["POST"])
def generate_visualizations():
    data = request.get_json()
    try:
        income = float(data.get("income", 0))
        credit_score = int(data.get("credit_score", 0))
        gpa = float(data.get("gpa", 0))
    except Exception:
        return jsonify({"error": "Invalid input data."}), 400

    expected_features = get_expected_features()
    input_dict = {key: value for key, value in {
        "income": income,
        "credit_score": credit_score,
        "gpa": gpa,
        "debt_to_income_ratio": 0.2,
        "household_income": income + 10000,
        "part_time_income": 3000,
        "financial_support": 10000,
        "student_debt": 15000,
        "tuition_cost": 25000
    }.items() if key in expected_features}
    
    input_data = pd.DataFrame([input_dict])
    X_scaled = scaler.transform(input_data)

    # Ensure static folder exists
    static_dir = os.path.join(os.getcwd(), "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # SHAP Visualization
    try:
        explainer = shap.TreeExplainer(dt_model)
        shap_values = explainer.shap_values(X_scaled)
        shap_plot_path = os.path.join(static_dir, "shap_summary_plot.png")
        shap.summary_plot(shap_values[1], X_scaled, feature_names=input_data.columns, show=False)
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Error generating SHAP plot:", e)
        shap_plot_path = ""

    # LIME Visualization
    try:
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=input_data.columns,
            mode="classification"
        )
        explanation = explainer_lime.explain_instance(X_scaled[0], dt_model.predict_proba, num_features=5)
        lime_plot_path = os.path.join(static_dir, "lime_explanation_plot.html")
        explanation.save_to_file(lime_plot_path)
    except Exception as e:
        print("Error generating LIME explanation:", e)
        lime_plot_path = ""

    # ROC Curve (Performance Metric)
    try:
        X_dummy, y_dummy = make_classification(n_samples=100, n_features=input_data.shape[1], n_classes=2, random_state=42)
        dummy_rf = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_rf.fit(X_dummy, y_dummy)
        y_probs = dummy_rf.predict_proba(X_dummy)[:, 1]
        fpr, tpr, _ = roc_curve(y_dummy, y_probs)
        roc_auc = auc(fpr, tpr)
        roc_plot_path = os.path.join(static_dir, "roc_curve.png")
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(roc_plot_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Error generating ROC curve:", e)
        roc_plot_path = ""

    return jsonify({
        "shap_plot": "/static/shap_summary_plot.png",
        "lime_plot": "/static/lime_explanation_plot.html",
        "roc_curve": "/static/roc_curve.png"
    })

if __name__ == "__main__":
    app.run(debug=True)
