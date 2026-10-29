import os
import pandas as pd
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier

app = Flask(__name__)

# ---------------------------
# Load Model
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "catboost_liver_model.cbm")

model = CatBoostClassifier(verbose=False)

if os.path.exists(model_path):
    model.load_model(model_path)
else:
    raise Exception("Model file not found!")

# ---------------------------
# Required Features
# ---------------------------
REQUIRED_FEATURES = [
    'age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status',
    'hepatitis_b', 'hepatitis_c', 'liver_function_score',
    'alpha_fetoprotein_level', 'cirrhosis_history',
    'family_history_cancer', 'physical_activity_level', 'diabetes'
]

# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def home():
    return jsonify({
        "message": "Liver Cancer Prediction API is running"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 1) Check JSON exists
        if not data:
            return jsonify({"error": "No JSON provided"}), 400

        # 2) Check required fields
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        # 3) Convert to DataFrame
        input_df = pd.DataFrame([data])

        # 4) Prediction
        proba = model.predict_proba(input_df)[0, 1]
        percentage = round(float(proba) * 100, 2)

        if percentage < 30:
            status = "Low Risk"
        elif percentage < 60:
            status = "Moderate Risk"
        else:
            status = "High Suspicion"

        # 5) Response
        return jsonify({
            "success": True,
            "probability": percentage,
            "status": status
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------------------
# Run
# ---------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)