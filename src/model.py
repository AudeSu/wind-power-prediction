from sklearn.ensemble import RandomForestRegressor
import joblib
from typing import Dict

class WindPowerModel:
    def __init__(self, params: Dict):
        self.model = RandomForestRegressor(**params)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str):
        model = cls({})
        model.model = joblib.load(path)
        return model