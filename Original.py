import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_prep import *
from functions import *


time = np.arange(4 * 365 + 1)


slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
noise_level = 5
noise = white_noise(time, noise_level, seed=42)
series += noise

plt.figure(figsize=(10, 6))
plot_series(btc_time, btc_data)
plt.show()


split_time = 1000


time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = seq2seq_window_dataset(x_train, window_size,
                                   batch_size=128)
valid_set = seq2seq_window_dataset(x_valid, window_size,
                                   batch_size=128)

""" Fully Convolutional Preprocessing 
    causal padding (add zeros at beginning to predict first instance
model = keras.models.Sequential([
  keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  keras.layers.LSTM(32, return_sequences=True),
  keras.layers.LSTM(32, return_sequences=True),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200)
])

lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
#history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
#plt.semilogx(history.history["lr"], history.history["loss"])
#plt.axis([1e-8, 1e-4, 0, 30])
#plt.show()

We know the apparent best lr, let's fit the model again with callbacks

optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_CNN_forecast_checkpoint.h5", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=50)
#history= model.fit(train_set, epochs=500,
#          validation_data=valid_set,
#          callbacks=[early_stopping, model_checkpoint])
"""
model = keras.models.load_model("my_CNN_forecast_checkpoint.h5")

rnn_forecast = model_forecast(model, series[:,  np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.title("rnn_forecast vs x_valid")
print(keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())
plt.show()


""" Let's move on to a CNN with WaveNet architecture, Fully Convolutional
    - NO LSTM
    - dilation rate for convolution"""


window_size = 64
train_set = seq2seq_window_dataset(x_train, window_size,
                                   batch_size=128)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, 1]))
for dilation_rate in (1, 2, 4, 8, 16, 32):
    model.add(
        #WaveNet architecture features
      keras.layers.Conv1D(filters=32,
                          kernel_size=2,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding="causal",
                          activation="relu")
    )
    #This layer acts as a Dense layer, taking all 32 previous input and returning 1 output, the actual pred.
model.add(keras.layers.Conv1D(filters=1, kernel_size=1))

wn_lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 30))

optimizer = keras.optimizers.Adam(lr=1e-4)

model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
#history = model.fit(train_set, epochs=100, callbacks=[wn_lr_schedule])
#plt.semilogx(history.history["lr"], history.history["loss"])
#plt.axis([1e-4, 1e-1, 0, 30])
#plt.show()


wn_optimizer = keras.optimizers.Adam(lr=3e-4)
model.compile(loss=keras.losses.Huber(),
              optimizer=wn_optimizer,
              metrics=["mae"])

model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_wn_cnn_checkpoint.h5", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=50)
#history = model.fit(train_set, epochs=500,
#                    validation_data=valid_set,
#                    callbacks=[early_stopping, model_checkpoint])

model = keras.models.load_model("my_wn_cnn_checkpoint.h5")

cnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
cnn_forecast = cnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, cnn_forecast)
plot_series(time_valid, rnn_forecast)

wn_cnn_mae = keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()
rnn_mae = keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
diff_mae = abs(wn_cnn_mae - rnn_mae)
print("difference mean absolute error between bw rnn {} and wn_cnn {} is : {}".format(rnn_mae, wn_cnn_mae, diff_mae ))
plt.show()