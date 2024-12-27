from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print("\nModel Performance:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R² Score: {r2:.6f}")
        
        return {'mse': mse, 'rmse': rmse, 'r2': r2}