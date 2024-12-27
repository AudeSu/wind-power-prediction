import pandas as pd
from typing import Tuple

class DataLoader:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and preprocess the wind power data."""
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
        
        if 'Unnamed: 0' in df.columns:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
            
        return df
    
    @staticmethod
    def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split features and target variable."""
        features = df.drop(columns=['Time', 'Power'])
        target = df['Power']
        return features, target