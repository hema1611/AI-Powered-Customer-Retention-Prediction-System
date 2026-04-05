from flask import Flask, render_template, request
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standar_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            form = request.form

            # ── 1. Numeric inputs ────────────────────────────────────────
            tenure          = float(form["tenure_trim"])
            monthly_charges = float(form["MonthlyCharges_trim"])
            total_charges   = float(form["TotalCharges_mode_trim"])

            # ── 2. Binary: Personal info ─────────────────────────────────
            gender_male    = 1 if form["gender"]        == "Male" else 0
            partner_yes    = 1 if form["Partner"]       == "Yes"  else 0
            dependents_yes = 1 if form["Dependents"]    == "Yes"  else 0
            phone_yes      = 1 if form["PhoneService"]  == "Yes"  else 0

            # ── 3. MultipleLines → 2 columns ─────────────────────────────
            ml = form["MultipleLines"]
            ml_no_phone = 1 if ml == "No phone service" else 0
            ml_yes      = 1 if ml == "Yes"              else 0

            # ── 4. InternetService → 2 columns (DSL = baseline) ──────────
            inet = form["InternetService"]
            inet_fiber = 1 if inet == "Fiber optic" else 0
            inet_no    = 1 if inet == "No"          else 0

            # ── 5. Service add-ons → each produces 2 columns ─────────────
            def encode_service(val):
                return (
                    1 if val == "No internet service" else 0,
                    1 if val == "Yes"                 else 0
                )

            os_ni,  os_yes  = encode_service(form["OnlineSecurity"])
            ob_ni,  ob_yes  = encode_service(form["OnlineBackup"])
            dp_ni,  dp_yes  = encode_service(form["DeviceProtection"])
            ts_ni,  ts_yes  = encode_service(form["TechSupport"])
            stv_ni, stv_yes = encode_service(form["StreamingTV"])
            sm_ni,  sm_yes  = encode_service(form["StreamingMovies"])

            # ── 6. Paperless Billing ──────────────────────────────────────
            paperless_yes = 1 if form["PaperlessBilling"] == "Yes" else 0

            # ── 7. PaymentMethod → 3 columns (Bank transfer = baseline) ───
            pm = form["PaymentMethod"]
            pm_cc     = 1 if pm == "Credit card (automatic)" else 0
            pm_echeck = 1 if pm == "Electronic check"        else 0
            pm_mail   = 1 if pm == "Mailed check"            else 0

            # ── 8. Network provider → 3 columns (Airtel = baseline) ───────
            np_ = form["network_provider"].lower()
            np_bsnl = 1 if np_ == "bsnl" else 0
            np_jio  = 1 if np_ == "jio"  else 0
            np_vi   = 1 if np_ == "vi"   else 0

            # ── 9. Contract → ordinal encoded ─────────────────────────────
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            contract_od  = contract_map.get(form["Contract"], 0)

            # ── Assemble in exact model column order (31 features) ─────────
            # ['tenure_trim', 'MonthlyCharges_trim', 'TotalCharges_mode_trim',
            #  'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
            #  'MultipleLines_No phone service', 'MultipleLines_Yes',
            #  'InternetService_Fiber optic', 'InternetService_No',
            #  'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            #  'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            #  'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            #  'TechSupport_No internet service', 'TechSupport_Yes',
            #  'StreamingTV_No internet service', 'StreamingTV_Yes',
            #  'StreamingMovies_No internet service', 'StreamingMovies_Yes',
            #  'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
            #  'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
            #  'network_provider_bsnl', 'network_provider_jio', 'network_provider_vi',
            #  'Contract_od']
            features = [
                tenure, monthly_charges, total_charges,
                gender_male, partner_yes, dependents_yes, phone_yes,
                ml_no_phone, ml_yes,
                inet_fiber, inet_no,
                os_ni,  os_yes,
                ob_ni,  ob_yes,
                dp_ni,  dp_yes,
                ts_ni,  ts_yes,
                stv_ni, stv_yes,
                sm_ni,  sm_yes,
                paperless_yes,
                pm_cc, pm_echeck, pm_mail,
                np_bsnl, np_jio, np_vi,
                contract_od
            ]

            # ── Scale & Predict ────────────────────────────────────────────
            features_array  = np.array([features])
            features_scaled = scaler.transform(features_array)
            pred            = model.predict(features_scaled)[0]

            prediction = "Churn" if pred == 1 else "No Churn"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)