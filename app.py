from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("Saved_Model/logistic_regression_model.pkl")
scaler = joblib.load("Saved_Model/logistic_regression_scaler.pkl")
feature_names = joblib.load("Saved_Model/feature_names.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API Running"}

@app.post("/predict")
def predict(customer : dict):
    df = pd.DataFrame([customer])

    df = df.reindex(columns=feature_names ,fill_value=0)

    scaled_data = scaler.transform(df)

    prediction = int(model.predict(scaled_data)[0])
    probability = float(model.predict_proba(scaled_data)[0][1])

    return{
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "result": "Customer Will Churn"
                  if prediction == 1
                  else "Customer Will Not Churn"
    }
'''
1-male
0-female
Request Format:
{
  "gender": 0,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 45,
  "PhoneService": 1,
  "PaperlessBilling": 0
}'''