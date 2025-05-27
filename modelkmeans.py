from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances

app = FastAPI(title="Protein Similarity Finder")
try:
    scaler = joblib.load("K-Means/scaler.pkl")
    kmeans = joblib.load("K-Means/kmeans.pkl")
    X_train = pd.read_csv("K-Means/train_data.csv")
except Exception as e:
    raise RuntimeError(f"Gagal load model atau data: {e}")

features = [
    'molecular_weight',
    'isoelectric_point',
    'hydrophobicity',
    'total_charge',
    'polar_ratio',
    'nonpolar_ratio',
    'sequence_length'
]

class FisiokimiaInput(BaseModel):
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity: float
    total_charge: int
    polar_ratio: float
    nonpolar_ratio: float
    sequence_length: int

@app.post("/find-similar/")
def find_similar(input_data: FisiokimiaInput):
    try:
        # Convert input ke vektor & scale
        user_vec = np.array([[getattr(input_data, f) for f in features]])
        user_vec_scaled = scaler.transform(user_vec)

        # Prediksi cluster
        user_cluster = int(kmeans.predict(user_vec_scaled)[0])

        # Ambil data dari cluster yang sama
        similar_df = X_train[X_train['cluster'] == user_cluster].copy()

        if similar_df.empty:
            raise HTTPException(status_code=404, detail="Tidak ada data serupa di cluster ini.")

        # Hitung jarak Euclidean
        distances = euclidean_distances(
            scaler.transform(similar_df[features]),
            user_vec_scaled
        )
        similar_df['distance'] = distances
        closest_match = similar_df.sort_values('distance').iloc[0]

        return {
            "sequence": closest_match['sequence'],
            "distance": round(float(closest_match['distance']), 4),
            "cluster": user_cluster
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
