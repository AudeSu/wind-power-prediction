# Wind Power Prediction Pipeline

This project implements a machine learning pipeline for predicting wind power generation based on various environmental features.

## Project Structure

```
wind_power_prediction/
│
├── data/                      # Data directory
├── notebooks/                 # Jupyter notebooks
├── src/                      # Source code
│   ├── data/                 # Data loading and preprocessing
│   ├── features/             # Feature engineering
│   ├── models/               # Model training and evaluation
│   └── visualization/        # Plotting and visualization
├── tests/                    # Unit tests
└── train.py                  # Main training script
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd wind_power_prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Place your training data in the `data/` directory.

## Usage

1. Train the models:
```bash
python train.py
```

2. The pipeline includes:
   - Data loading and preprocessing
   - Feature engineering and scaling
   - Model training (Linear Regression, Random Forest, Gradient Boosting)
   - Ensemble model creation
   - Model evaluation and visualization

## Models

The pipeline implements multiple models:
- Linear Regression (baseline)
- Random Forest
- Gradient Boosting
- Ensemble (weighted combination of RF and GB)

## Features

Key features used for prediction:
- Wind speed at different heights
- Temperature
- Relative humidity
- Dew point
- Wind direction
- Wind gust

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
