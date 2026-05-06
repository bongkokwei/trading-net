"""Neural network models for time-series prediction."""

from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization


def model_2layer(X, Y, num_feature=30, l2_reg=0.001, drop_prop=0.25,
                 num_epoch=1, batch_s=None):
    """
    Build and train a 2-layer neural network.

    Architecture: RELU -> RELU -> LINEAR

    Args:
        X: Training features
        Y: Training targets
        num_feature: Input feature dimension
        l2_reg: L2 regularization coefficient
        drop_prop: Dropout proportion
        num_epoch: Number of training epochs
        batch_s: Batch size

    Returns:
        Trained Keras model
    """
    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_shape=(num_feature,)))
    model.add(ActivityRegularization(l2=l2_reg))
    model.add(Dropout(drop_prop))

    model.add(Dense(units=250, activation='relu'))
    model.add(ActivityRegularization(l2=l2_reg))
    model.add(Dropout(drop_prop))

    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, Y, epochs=num_epoch, batch_size=batch_s)

    return model


def predict_2layer(model, X):
    """Generate predictions from trained model."""
    return model.predict(X)
