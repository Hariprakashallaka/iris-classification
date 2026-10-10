
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Define the model path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"

# 2. Load the trained model
model = joblib.load(MODEL_PATH)

# 3. Create the FastAPI application
app = FastAPI(
    title="Iris Flower Classification API",
    description="Predict Iris flower species from four measurements.",
    version="1.0.0"
)

# 4. Define the input data structure
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# 5. Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris Classification API is running",
        "docs": "/docs"
    }

# 6. Health-check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}

# 7. Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    input_df = pd.DataFrame([{
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width
    }])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = float(probabilities.max())

    return {
        "predicted_species": str(prediction),
        "confidence": round(confidence, 4)
    }