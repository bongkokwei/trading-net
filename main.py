from trading_util import *
from keras_NN import *

#initialise hyperparater
train_day = 20
predt_day = 1
steps_day = 1
mb_size = 128 # mini-batch

data = read_file('sp500.csv')
data_train =  data_query(data, '1996-01-04', '2016-12-30')
X_train, Y_train = split_timeseries_v2(data_train['close'], train_day, predt_day, steps_day)
data_test = data_query(data, '2017-01-03', '2017-12-29')
X_test, Y_test = split_timeseries_v2(data_test['close'], train_day, predt_day, steps_day)

model = model_2layer(X_train, Y_train, num_feature = train_day, num_epoch = 70, batch_s = mb_size, drop_prop = 0, l2_reg = 0.0008)
score = model.evaluate(X_test, Y_test, batch_size=mb_size)
print(score)

# draw graph, red-test, blue-predict
Y_predict = predict_2layer(model, X_test)
plot_result(Y_test,Y_predict)
