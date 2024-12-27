from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.config import Config
from src.model import WindPowerModel

app = FastAPI(title="Wind Power Prediction API")

class WindData(BaseModel):
    Location: float
    Temp_2m: float
    RelHum_2m: float
    DP_2m: float
    WS_10m: float
    WS_100m: float
    WD_10m: float
    WD_100m: float
    WG_10m: float

@app.post("/predict")
async def predict(data: WindData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Load model and make prediction
        model = WindPowerModel.load(str(Config.MODEL_DIR / "rf_model.pkl"))
        prediction = model.predict(input_df)[0]
        
        return {"predicted_power": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}