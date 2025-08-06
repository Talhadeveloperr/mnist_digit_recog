# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('../model/best_model.pkl')
scaler = joblib.load('../model/scaler.pkl')

class DigitInput(BaseModel):
    pixels: list

@app.post("/predict")
def predict_digit(data: DigitInput):
    x = np.array(data.pixels).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)
    return {"prediction": int(prediction[0])}
