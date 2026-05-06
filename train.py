"""Main training script for S&P 500 price prediction."""

from tradingnet.utils import read_file, data_query, split_timeseries_v2, plot_result
from tradingnet.model import model_2layer, predict_2layer

# Hyperparameters
TRAIN_DAYS = 20
PREDICT_DAYS = 1
STEP_SIZE = 1
BATCH_SIZE = 128

# Load and prepare data
data = read_file('data/sp500.csv')

data_train = data_query(data, '1996-01-04', '2016-12-30')
X_train, Y_train = split_timeseries_v2(
    data_train['close'], TRAIN_DAYS, PREDICT_DAYS, STEP_SIZE
)

data_test = data_query(data, '2017-01-03', '2017-12-29')
X_test, Y_test = split_timeseries_v2(
    data_test['close'], TRAIN_DAYS, PREDICT_DAYS, STEP_SIZE
)

# Train model
model = model_2layer(
    X_train, Y_train,
    num_feature=TRAIN_DAYS,
    num_epoch=70,
    batch_s=BATCH_SIZE,
    drop_prop=0,
    l2_reg=0.0008
)

# Evaluate on test set
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
print(f"Test loss: {score}")

# Visualize predictions
Y_predict = predict_2layer(model, X_test)
plot_result(Y_test, Y_predict)
