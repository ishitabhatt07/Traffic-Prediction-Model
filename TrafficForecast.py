import numpy as np
import pandas as pd
import math
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

# Fixing random seed for reproducibility
np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)

# Read data from files to dataframes
df1 = pd.read_csv("./Traffic_Data_cali/train.csv", encoding='utf-8')
df2 = pd.read_csv("./Traffic_Data_cali/test.csv", encoding='utf-8')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1['Flow (Veh/5 Minutes)'].values.reshape(-1, 1))
train_data = scaler.transform(df1['Flow (Veh/5 Minutes)'].values.reshape(-1, 1)).reshape(1, -1)[0]
test_data = scaler.transform(df2['Flow (Veh/5 Minutes)'].values.reshape(-1, 1)).reshape(1, -1)[0]

# Practice with different time lag (look back) values to optimize the models
lag = 12
train, test = [], []
for i in range(lag, len(train_data)):
    train.append(train_data[i - lag: i + 1])
for i in range(lag, len(test_data)):
    test.append(test_data[i - lag: i + 1])

train = np.array(train)
test = np.array(test)

# Shuffle data (stateless case)
np.random.shuffle(train)
x_train = train[:, :-1]
y_train = train[:, -1]
x_test = test[:, :-1]
y_test = test[:, -1]


# Building models
def build_LSTM():
    model = Sequential()
    model.add(LSTM(64, input_shape=(lag, 1), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_GRU():
    model = Sequential()
    model.add(GRU(64, input_shape=(lag, 1), return_sequences=True))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


# Model structure
model_struct = "GRU"  # or "LSTM"

# Train the models
# if model_struct == "LSTM":
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     model = build_LSTM()
#     model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
#     hist = model.fit(x_train, y_train, batch_size=64, epochs=600, validation_split=0.05)
#     model.save('models/LSTM.h5')
#     df = pd.DataFrame.from_dict(hist.history)
#     df.to_csv('models/LSTM_loss.csv', encoding='utf-8', index=False)
# elif model_struct == "GRU":
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     model = build_GRU()
#     model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
#     hist = model.fit(x_train, y_train, batch_size=64, epochs=600, validation_split=0.05)
#     model.save('models/GRU.h5')
#     df = pd.DataFrame.from_dict(hist.history)
#     df.to_csv('models/GRU_loss.csv', encoding='utf-8', index=False)


# Evaluate models and plot graphs

# Calculate Mean Absolute Percentage Error
def evaluate_models(y_true, y_pred):
    y_true = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    sums = 0
    for i in range(len(y_pred)):
        tmp = abs(y_true[i] - y_pred[i]) / y_true[i]
        sums += tmp
    mape = sums * (100 / len(y_pred))

    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance_score: %f' % vs)
    print('mape: %f%%' % mape)
    print('mae: %f' % mae)
    print('mse: %f' % mse)
    print('rmse: %f' % math.sqrt(mse))
    print('r2: %f' % r2)


# Plot one-day predictions for LSTM
def plot_LSTM_oneDay(y_true, y_pred):
    d = '2020-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred[:288], label='LSTM')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Flow')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()


# Plot one-day predictions for GRU
def plot_GRU_oneDay(y_true, y_pred):
    d = '2020-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred[:288], label='GRU')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Flow')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()


# Plot one-day predictions for both LSTM and GRU
def plot_models_oneDay(y_true, y_pred):
    d = '2020-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred[0][:288], label='LSTM')
    ax.plot(x, y_pred[1][:288], label='GRU')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Flow')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()


# Plot one-week predictions for both LSTM and GRU
def plot_models_oneWeek(y_true, y_pred):
    x = pd.date_range(start='2020-3-4 00:00', end='2020-3-11 00:00', freq='5min')
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred[0][:2017], label='LSTM')
    ax.plot(x, y_pred[1][:2017], label='GRU')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Flow')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()


# Load models
lstm = load_model('./models/LSTM.h5')
gru = load_model('./models/GRU.h5')
models = [lstm, gru]

# Evaluate models
names = ['LSTM', 'GRU']
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
y_preds = []
for name, model in zip(names, models):
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted = model.predict(x_test)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
    y_preds.append(predicted)
    print(name)
    evaluate_models(y_test, predicted)


# Plot graphs
plot_LSTM_oneDay(y_test[:288], y_preds[0])  # Pass only LSTM predictions
plot_GRU_oneDay(y_test[:288], y_preds[1])   # Pass only GRU predictions
plot_models_oneDay(y_test[:288], y_preds)   # Pass both LSTM and GRU predictions
plot_models_oneWeek(y_test[:2017], y_preds) # Pass both LSTM and GRU predictions
