import os
from pathlib import Path

class Config:
    # Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Model hyperparameters
    RF_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    GB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': RANDOM_STATE
    }
    
    # Ensemble weights
    RF_WEIGHT = 0.6
    GB_WEIGHT = 0.4