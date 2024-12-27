from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from typing import Dict, Any

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any] = None):
        """Create and return a model instance."""
        if model_type == "rf":
            return RandomForestRegressor(**(params or {}))
        elif model_type == "gb":
            return GradientBoostingRegressor(**(params or {}))
        elif model_type == "lr":
            return LinearRegression(**(params or {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_ensemble_prediction(rf_pred: np.ndarray, gb_pred: np.ndarray, 
                                 rf_weight: float, gb_weight: float) -> np.ndarray:
        """Create ensemble prediction from RF and GB predictions."""
        return (rf_weight * rf_pred) + (gb_weight * gb_pred)
    
    @staticmethod
    def save_models(models: Dict[str, Any], base_path: str):
        """Save trained models to disk."""
        for name, model in models.items():
            joblib.dump(model, f"{base_path}/{name}_model.pkl")
    
    @staticmethod
    def load_models(base_path: str) -> Dict[str, Any]:
        """Load trained models from disk."""
        model_files = {
            'lr': f"{base_path}/lr_model.pkl",
            'rf': f"{base_path}/rf_model.pkl",
            'gb': f"{base_path}/gb_model.pkl"
        }
        
        models = {}
        for name, path in model_files.items():
            try:
                models[name] = joblib.load(path)
            except FileNotFoundError:
                print(f"Warning: Model {name} not found at {path}")
        return models