from pathlib import Path

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    RF_PARAMS = {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': RANDOM_STATE
    }