# Trading-Net - Project Documentation

## Overview
Time-series prediction model for S&P 500 stock prices using a 2-layer neural network.

## Project Structure
- **tradingnet/**: Main package containing models and utilities
  - `model.py`: Neural network architecture and training functions
  - `utils.py`: Data loading, processing, and visualization
- **data/**: CSV and pickle data files
- **tests/**: Unit tests (to be added)
- **train.py**: Main training script

## Key Components

### Data Pipeline
1. `read_file()`: Loads CSV data and caches as pickle for speed
2. `data_query()`: Filters data by date range
3. `split_timeseries_v2()`: Creates training windows with global normalization

### Model
- 2-layer sequential model with RELU activations
- L2 regularization and dropout for prevention of overfitting
- MSE loss with Adam optimizer
- Trained on 1996-2016 data, tested on 2017 data

## Recent Changes (Refactoring)
- Organized files into proper package structure
- Created `tradingnet/` package with separate `model.py` and `utils.py`
- Moved data to `data/` directory
- Added `requirements.txt` for dependency management
- Added `.gitignore` to exclude cache files
- Created `setup.py` for package installation
- Renamed `main.py` to `train.py` for clarity
- Fixed code quality issues:
  - Fixed `plt.show()` call (was missing parentheses)
  - Improved docstrings and code comments
  - Removed incomplete/unclear comments

## Dependencies
See `requirements.txt` - main deps are TensorFlow/Keras, scikit-learn, numpy, matplotlib

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Future Enhancements
- Add LSTM/GRU models
- Implement proper train/val/test split with cross-validation
- Add model checkpointing and saving
- Add unit tests in `tests/` directory
- Support multiple stock symbols
- Add more technical indicators
