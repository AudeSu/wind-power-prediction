from src.config import Config
from src.data_loader import DataLoader
from src.model import WindPowerModel
from src.evaluation import ModelEvaluator
from sklearn.model_selection import train_test_split
import os

def main():
    # Create model directory
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load and prepare data
    data_loader = DataLoader()
    df = data_loader.load_data(str(Config.DATA_DIR / "Train.csv"))
    X, y = data_loader.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # Train model
    model = WindPowerModel(Config.RF_PARAMS)
    model.train(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    ModelEvaluator.evaluate_model(y_test, predictions)
    
    # Save model
    model.save(str(Config.MODEL_DIR / "rf_model.pkl"))

if __name__ == "__main__":
    main()