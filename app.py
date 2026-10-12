from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap
import uvicorn
import os

app = FastAPI(title="XAI Medical Diagnosis System")

# Load model artifacts
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)
    
model = artifacts["model"]
features = artifacts["features"]
classes = artifacts["classes"]

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

class PatientData(BaseModel):
    Age: float
    BMI: float
    Glucose: float
    Blood_Pressure: float
    Cholesterol: float
    Heart_Rate: float

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to dataframe
    input_data = pd.DataFrame([[
        data.Age, data.BMI, data.Glucose, data.Blood_Pressure, 
        data.Cholesterol, data.Heart_Rate
    ]], columns=features)
    
    # Get prediction
    pred_class = model.predict(input_data)[0]
    
    # Get probabilities
    probs = model.predict_proba(input_data)[0]
    confidence = max(probs)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)
    
    # In SHAP 0.45+, TreeExplainer.shap_values returns an array of shape (n_samples, n_features, n_classes)
    # for scikit-learn Random Forest multiclass
    class_idx = np.where(model.classes_ == pred_class)[0][0]
    
    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_idx][0]
    elif len(shap_values.shape) == 3:
        class_shap_values = shap_values[0, :, class_idx]
    else:
        class_shap_values = shap_values[0]
            
    # Format SHAP values for the frontend
    feature_contributions = []
    for i, feature in enumerate(features):
        feature_contributions.append({
            "feature": feature.replace('_', ' '),
            "value": float(class_shap_values[i]),
            "input_value": float(input_data.iloc[0][feature])
        })
        
    # Sort by absolute contribution
    feature_contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
    
    # Generate simple explanation
    top_feature = feature_contributions[0]
    second_feature = feature_contributions[1]
    
    # Add context to explanation
    if pred_class == "Healthy":
        explanation = f"The model predicts you are generally healthy. Your {top_feature['feature']} ({top_feature['input_value']}) and {second_feature['feature']} ({second_feature['input_value']}) strongly support this positive outcome."
        risk_level = "Low"
    else:
        explanation = f"The model detected a risk of {pred_class}. The primary contributing factors increasing your risk are your {top_feature['feature']} and {second_feature['feature']}."
        risk_level = "High" if confidence > 0.7 else "Moderate"
    
    return {
        "prediction": pred_class,
        "confidence": float(confidence),
        "risk_level": risk_level,
        "contributions": feature_contributions,
        "explanation": explanation
    }

# Create static dir if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8555, reload=True)
