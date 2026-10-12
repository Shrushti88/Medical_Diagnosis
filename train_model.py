import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate base features
    age = np.random.randint(20, 80, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    glucose = np.random.normal(100, 30, n_samples)
    blood_pressure = np.random.normal(120, 15, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    heart_rate = np.random.normal(70, 10, n_samples)
    
    # Clip values to realistic ranges
    bmi = np.clip(bmi, 15, 50)
    glucose = np.clip(glucose, 60, 250)
    blood_pressure = np.clip(blood_pressure, 80, 200)
    cholesterol = np.clip(cholesterol, 100, 350)
    heart_rate = np.clip(heart_rate, 40, 120)
    
    # Create target labels based on rules (with some noise)
    labels = []
    for i in range(n_samples):
        # Calculate risk scores
        diabetes_risk = (glucose[i] > 125) * 2 + (bmi[i] > 30) * 1 + (age[i] > 50) * 0.5
        heart_risk = (blood_pressure[i] > 140) * 2 + (cholesterol[i] > 240) * 1.5 + (age[i] > 60) * 1
        
        # Add random noise
        diabetes_risk += np.random.normal(0, 0.5)
        heart_risk += np.random.normal(0, 0.5)
        
        if diabetes_risk > 2.5 and diabetes_risk > heart_risk:
            labels.append("Diabetes Risk")
        elif heart_risk > 3.0:
            labels.append("Heart Disease Risk")
        else:
            labels.append("Healthy")
            
    data = pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'Glucose': glucose,
        'Blood_Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Heart_Rate': heart_rate,
        'Disease': labels
    })
    
    return data

def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data(2000)
    
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Model Accuracy: {score:.2f}")
    
    # Save model and feature names
    print("Saving model artifacts...")
    artifacts = {
        'model': model,
        'features': list(X.columns),
        'classes': list(model.classes_)
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
        
    print("Training complete! Model saved to model.pkl")

if __name__ == "__main__":
    main()
