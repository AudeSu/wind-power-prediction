import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

class Visualizer:
    @staticmethod
    def plot_power_distribution(df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Power'], kde=True, bins=30)
        plt.title('Distribution of Power Generation')
        plt.xlabel('Power')
        plt.ylabel('Frequency')
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame):
        plt.figure(figsize=(12, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix of Features')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(models: List[str], metrics: List[float], 
                            metric_name: str, color: str = 'skyblue'):
        plt.figure(figsize=(10, 6))
        plt.bar(models, metrics, color=color)
        plt.title(f'Model Comparison - {metric_name}')
        plt.ylabel(metric_name)
        plt.show()