from src.config import Config
from src.data_loader import DataLoader
from src.model import WindPowerModel
import pandas as pd

def predict_power(input_data: pd.DataFrame) -> float:
    """Make wind power prediction using the trained model."""
    # Load model
    model = WindPowerModel.load(str(Config.MODEL_DIR / "rf_model.pkl"))
    
    # Prepare input data
    if 'Time' in input_data.columns:
        input_data = input_data.drop('Time', axis=1)
    if 'Power' in input_data.columns:
        input_data = input_data.drop('Power', axis=1)
    
    # Make prediction
    return model.predict(input_data)

if __name__ == "__main__":
    # Example prediction
    sample_input = pd.DataFrame({
        'Location': [1],
        'Temp_2m': [28.17960022],
        'RelHum_2m': [85.66420479],
        'DP_2m': [24.27259522],
        'WS_10m': [2.225389169],
        'WS_100m': [3.997799417],
        'WD_10m': [150.0516826],
        'WD_100m': [157.0573147],
        'WG_10m': [4.336515074]
    })
    
    prediction = predict_power(sample_input)
    print(f"\nPredicted Power: {prediction[0]}")