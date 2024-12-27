from src.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
import pandas as pd

def load_pipeline_components():
    """Load all necessary pipeline components."""
    # Load models
    models = ModelFactory.load_models(str(Config.MODEL_DIR))
    
    # Load feature engineer
    feature_engineer = FeatureEngineer.load(str(Config.MODEL_DIR / "feature_engineer.pkl"))
    
    return models, feature_engineer

def predict_power(input_data: pd.DataFrame, model_type: str = 'ensemble'):
    """
    Make predictions using the trained models.
    
    Args:
        input_data: DataFrame with the same features as training data
        model_type: Type of model to use ('lr', 'rf', 'gb', or 'ensemble')
    
    Returns:
        Predicted power values
    """
    # Load models and feature engineer
    models, feature_engineer = load_pipeline_components()
    
    # Preprocess input data
    if 'Time' in input_data.columns:
        input_data = input_data.drop('Time', axis=1)
    if 'Power' in input_data.columns:
        input_data = input_data.drop('Power', axis=1)
    
    # Scale features
    input_scaled = feature_engineer.transform(input_data)
    
    if model_type == 'ensemble':
        # Make predictions with RF and GB models
        rf_pred = models['rf'].predict(input_data)
        gb_pred = models['gb'].predict(input_data)
        
        # Create ensemble prediction
        predictions = ModelFactory.create_ensemble_prediction(
            rf_pred, gb_pred,
            Config.RF_WEIGHT, Config.GB_WEIGHT
        )
    else:
        if model_type == 'lr':
            input_data = input_scaled
        predictions = models[model_type].predict(input_data)
    
    return predictions

# Example usage:
if __name__ == "__main__":
    # Example for making predictions
    sample_input = pd.DataFrame({
        'Location': [1],
        'Temp_2m': [28.2796],
        'RelHum_2m': [84.664205],
        'DP_2m': [24.072595],
        'WS_10m': [1.605389],
        'WS_100m': [1.267799],
        'WD_10m': [145.051683],
        'WD_100m': [161.057315],
        'WG_10m': [1.336515]
    })
    
    # Make predictions using different models
    for model_type in ['lr', 'rf', 'gb', 'ensemble']:
        prediction = predict_power(sample_input, model_type)
        print(f"\n{model_type.upper()} Model Prediction:")
        print(prediction[0])