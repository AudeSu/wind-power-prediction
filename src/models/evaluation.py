from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from typing import Dict

class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics for model predictions."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str):
        """Print formatted metrics."""
        print(f"\n{model_name} Results:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"R² Score: {metrics['r2']:.6f}")