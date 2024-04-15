import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from functions import plot_series, seq2seq_window_dataset, model_forecast

keras = tf.keras

btc_data = pd.read_csv("../BTC Time Series/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
btc_data = btc_data[pd.notnull(btc_data['Weighted_Price'])]
#print(btc_data.describe())

#1.1 Bitcoin local csv
dates = pd.to_datetime(btc_data['Timestamp'])
btc_data = btc_data['Weighted_Price']


print(btc_data.head())

log_series = np.log10(btc_data)
time = np.arange(len(btc_data))

plt.figure(figsize=(10, 6))
plot_series(time, log_series)
#plt.show()


split_time = int(len(btc_data)*0.667)
print(split_time)

time_train = time[:split_time]
x_train = log_series[:split_time]
time_valid = time[split_time:]
x_valid = log_series[split_time:]


window_size = 64
valid_set = seq2seq_window_dataset(x_valid, window_size,
                                   batch_size=256)


window_size = 64
train_set = seq2seq_window_dataset(x_train, window_size,
                                   batch_size=256)

wavenet = keras.models.Sequential()
wavenet.add(keras.layers.InputLayer(input_shape=[None, 1]))
for dilation_rate in (1, 2, 4, 8, 16, 32):
    wavenet.add(
        #WaveNet architecture features
      keras.layers.Conv1D(filters=32,
                          kernel_size=2,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding="causal",
                          activation="relu")
    )
    #This layer acts as a Dense layer, taking all 32 previous input and returning 1 output, the actual pred.
wavenet.add(keras.layers.Conv1D(filters=1, kernel_size=1))

wn_lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 30))

optimizer = keras.optimizers.Adam(lr=1e-4)

wavenet.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

history = wavenet.fit(train_set, epochs=50, callbacks=[wn_lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-4, 1e-1, 0, 30])
plt.show()


wn_optimizer = keras.optimizers.Adam(lr=3e-4)
wavenet.compile(loss=keras.losses.Huber(),
              optimizer=wn_optimizer,
              metrics=["mae"])

model_checkpoint = keras.callbacks.ModelCheckpoint(
    "btc_wavenet.h5", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=50)
history = wavenet.fit(train_set, epochs=500,
                    validation_data=valid_set,
                    callbacks=[early_stopping, model_checkpoint])

model = keras.models.load_model("btc_wavenet.h5")
forecast = model_forecast(model, log_series[..., np.newaxis], window_size)
btc_wavenet_mea = keras.metrics.mean_absolute_error(x_valid, forecast).numpy()