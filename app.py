from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
with open("model_combined.pkl", "rb") as f:
    model = joblib.load("model_combined.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    form_data = {}

    if request.method == "POST":
        # store form values
        form_data = request.form.to_dict()

        features = [
            float(form_data["age"]),
            float(form_data["sex"]),
            float(form_data["chest_pain_type"]),
            float(form_data["resting_bp_s"]),
            float(form_data["cholesterol"]),
            float(form_data["fasting_blood_sugar"]),
            float(form_data["resting_ecg"]),
            float(form_data["max_heart_rate"]),
            float(form_data["exercise_angina"]),
            float(form_data["oldpeak"]),
            float(form_data["st_slope"]),
        ]

        features_array = np.array(features).reshape(1, -1)

        pred = model.predict(features_array)[0]
        prob = model.predict_proba(features_array)[0][1]

        prediction = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
        probability = round(prob * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        form_data=form_data
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

