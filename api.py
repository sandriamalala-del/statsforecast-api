from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

app = FastAPI(title="StatsForecast API for n8n")

class ForecastRequest(BaseModel):
    data: List[dict]
    h: int = 7
    freq: str = "D"

@app.get("/health")
def health():
    return {"status": "ok", "model": "AutoARIMA"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        # Convertit en DataFrame
        df = pd.DataFrame(request.data)
        if 'ds' not in df.columns or 'y' not in df.columns:
            return {"error": "Missing 'ds' or 'y' in data"}
        
        df = df[['ds', 'y']].copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['unique_id'] = 'series_1'
        
        # Modèle
        model = StatsForecast(models=[AutoARIMA(season_length=7)], freq=request.freq)
        
        # --- CORRECTION ---
        # On spécifie explicitement que 'df' est l'argument df
        fcst = model.forecast(df=df, h=request.h)
        # ------------------
        
        # Format sortie
        result = fcst.reset_index().to_dict(orient='records')
        return {"forecast": result}
    
    except Exception as e:
        return {"error": str(e)}
