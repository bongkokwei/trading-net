# Trading-Net

Exploring the nuances of predicting financial time-series data using deep learning.

## Overview

This project uses a 2-layer neural network (with RELU activations and dropout regularization) to predict S&P 500 closing prices based on historical time-series windows.

## Project Structure

```
trading-net/
├── tradingnet/          # Main package
│   ├── __init__.py
│   ├── model.py         # Neural network models
│   └── utils.py         # Data loading and processing utilities
├── data/                # Data files
│   └── sp500.csv        # S&P 500 historical data
├── tests/               # Unit tests
├── train.py             # Main training script
├── setup.py             # Package setup
├── requirements.txt     # Dependencies
└── README.md
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode (optional):
   ```bash
   pip install -e .
   ```

## Usage

Run the training script:

```bash
python train.py
```

This will:
1. Load S&P 500 historical data
2. Split data into training (1996-2016) and testing (2017) sets
3. Train a 2-layer neural network
4. Evaluate on test set
5. Visualize predictions vs actual prices

### Hyperparameters

Configure training parameters in `train.py`:
- `TRAIN_DAYS`: Number of days to use for each training window (default: 20)
- `PREDICT_DAYS`: Number of days to predict (default: 1)
- `STEP_SIZE`: Step size for sliding window (default: 1)
- `BATCH_SIZE`: Mini-batch size (default: 128)

## Model Architecture

The 2-layer model uses:
- **Input Layer**: 20 features (configurable)
- **Hidden Layer 1**: 500 units + RELU + L2 regularization + Dropout
- **Hidden Layer 2**: 250 units + RELU + L2 regularization + Dropout
- **Output Layer**: 1 unit + Linear activation (regression)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam

## Data Processing

The project includes two time-series splitting approaches:

- `split_timeseries()`: Normalizes data per sliding window
- `split_timeseries_v2()`: Normalizes entire dataset globally (recommended)

## Future Improvements

- Add more sophisticated architectures (LSTM, GRU)
- Implement cross-validation
- Add more technical indicators
- Support multiple stock symbols
- Add model persistence (save/load trained models)
- Expand test coverage

## License

MIT License

## References

- Keras/TensorFlow documentation
- Time-series forecasting best practices
