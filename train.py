from src.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.models.evaluation import ModelEvaluator
from src.visualization.plots import Visualizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def main():
    # Create necessary directories
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_data(str(Config.DATA_DIR / "Train.csv"))
    features, target = data_loader.split_features_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    X_train_scaled = feature_engineer.fit_transform(X_train)
    X_test_scaled = feature_engineer.transform(X_test)
    
    # Save feature engineer
    feature_engineer.save(str(Config.MODEL_DIR / "feature_engineer.pkl"))
    
    # Train models
    models = {}
    predictions = {}
    
    # Linear Regression
    models['lr'] = ModelFactory.create_model('lr')
    models['lr'].fit(X_train_scaled, y_train)
    predictions['lr'] = models['lr'].predict(X_test_scaled)
    
    # Random Forest
    models['rf'] = ModelFactory.create_model('rf', {'random_state': Config.RANDOM_STATE})
    models['rf'].fit(X_train, y_train)
    predictions['rf'] = models['rf'].predict(X_test)
    
    # Get feature importances and select top features
    feature_importances = pd.Series(
        models['rf'].feature_importances_, 
        index=features.columns
    ).sort_values(ascending=False)
    
    feature_engineer.set_top_features(feature_importances)
    
    # Gradient Boosting
    models['gb'] = ModelFactory.create_model('gb', Config.GB_PARAMS)
    models['gb'].fit(X_train, y_train)
    predictions['gb'] = models['gb'].predict(X_test)
    
    # Create ensemble predictions
    predictions['ensemble'] = ModelFactory.create_ensemble_prediction(
        predictions['rf'], predictions['gb'],
        Config.RF_WEIGHT, Config.GB_WEIGHT
    )
    
    # Save models
    ModelFactory.save_models(models, str(Config.MODEL_DIR))
    
    # Evaluate all models
    evaluator = ModelEvaluator()
    for model_name, y_pred in predictions.items():
        metrics = evaluator.evaluate_model(y_test, y_pred)
        evaluator.print_metrics(metrics, model_name.upper())

if __name__ == "__main__":
    main()