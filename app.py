from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap
import uvicorn
import os
from pathlib import Path

app = FastAPI(title="XAI Medical Diagnosis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths safely for local and serverless environments
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR.parent / "model.pkl"

STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = BASE_DIR.parent / "static"

# Load model artifacts
with open(MODEL_PATH, "rb") as f:
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

# Mount static files safely if directory exists
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def read_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    root_index = BASE_DIR / "index.html"
    if root_index.is_file():
        return FileResponse(str(root_index))
    return {"message": "XAI Medical Diagnosis System API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8555, reload=True)
