from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.top_features = None
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale features and select top important ones."""
        return self.scaler.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        return self.scaler.transform(X)
    
    def set_top_features(self, feature_importances: pd.Series, n_features: int = 5):
        """Set top important features based on model importance scores."""
        self.top_features = feature_importances.nlargest(n_features).index.tolist()
    
    def select_top_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only top important features."""
        if self.top_features is None:
            raise ValueError("Top features not set. Call set_top_features first.")
        return X[self.top_features]
    
    def save(self, path: str):
        """Save the feature engineer to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Load the feature engineer from disk."""
        return joblib.load(path)