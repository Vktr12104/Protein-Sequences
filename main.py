from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


model = tf.keras.models.load_model('model/lstm_model.keras')

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)

with open('model/config.json', 'r') as f:
    config = json.load(f)
max_length = config["max_length"]

output_labels = [
    'molecular_weight', 'isoelectric_point', 'hydrophobicity', 
    'total_charge', 'polar_ratio', 'nonpolar_ratio', 'sequence_length'
]
valid_chars = set(char_to_index.keys())

class SequenceInput(BaseModel):
    sequence: str

@app.get("/")
async def root():
    return {"message": "Protein property prediction API is running"}

@app.post("/predict")
async def predict(data: SequenceInput):
    try:
        seq = data.sequence.strip().upper()
        if not all(c in valid_chars for c in seq):
            raise HTTPException(status_code=400, detail="Sequence contains invalid characters")
        encoded = [char_to_index.get(c, 0) for c in seq]
        padded = pad_sequences([encoded], maxlen=max_length, padding='post')
        prediction_scaled = model.predict(padded)
        prediction = scaler.inverse_transform(prediction_scaled)
        result = {label: round(float(pred), 4) for label, pred in zip(output_labels, prediction[0])}
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
