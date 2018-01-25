from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization
from keras import optimizers

def model_2layer(X, Y, num_feature = 30, l2_reg = 0.001, drop_prop = 0.25,
                 num_epoch = 1, batch_s = None):
    """
    RELU -> RELU -> LINEAR
    """

    model = Sequential()
    # Hidden layer 1
    model.add(Dense(units=500, activation='relu', input_shape = (num_feature,)))
    model.add(ActivityRegularization(l2=l2_reg))
    model.add(Dropout(drop_prop))
    # Hidden layer 2
    model.add(Dense(units=250, activation='relu'))
    model.add(ActivityRegularization(l2=l2_reg))
    model.add(Dropout(drop_prop))
    # Output layer
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    # train model
    model.fit(X, Y, epochs=num_epoch, batch_size=batch_s)

    return model

def predict_2layer(model, X):

    Y_predict = model.predict(X)
    return Y_predict

#metrics module
