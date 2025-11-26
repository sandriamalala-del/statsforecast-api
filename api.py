from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
# Import de la librairie Prophet
from prophet import Prophet 

app = FastAPI(title="Prophet API for n8n")

class ForecastRequest(BaseModel):
    data: List[dict]
    h: int = 6
    freq: str = "W"

@app.get("/health")
def health():
    return {"status": "ok", "model": "Prophet"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        df = pd.DataFrame(request.data)
        if 'ds' not in df.columns or 'y' not in df.columns:
            return {"error": "Missing 'ds' or 'y' in data"}
        
        # Prophet est strict sur les noms ds/y
        df = df[['ds', 'y']].copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # --- LOGIQUE PROPHET ---
        m = Prophet()
        m.fit(df) # Entraînement
        
        # --- CORRECTION ICI ---
        # On ajoute freq=request.freq pour que le modèle respecte le "W" (hebdomadaire)
        future = m.make_future_dataframe(periods=request.h, freq=request.freq)
        # ----------------------
        
        # Prédiction
        forecast = m.predict(future)
        
        # On ne garde que les futures prédictions (les dernières lignes)
        result = forecast.tail(request.h)[['ds', 'yhat']].rename(columns={'yhat': 'Prophet'}).to_dict(orient='records')
        
        return {"forecast": result}
    
    except Exception as e:
        return {"error": str(e)}
