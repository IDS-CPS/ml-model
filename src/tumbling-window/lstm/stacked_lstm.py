import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense

## Plot the training history
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.savefig("plot/lstm-history.png")

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)

args = parser.parse_args()

df = pd.read_csv("dataset/swat-2015-data.csv", delimiter=";", decimal=",")
df = df[16000:]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)

# Subsample every 5 seconds
df = df[::5]

features_dropped = ["AIT201", "AIT202", "AIT203", "P201", "AIT401",
"AIT402", "AIT501", "AIT502", 'AIT503', "AIT504", "FIT503", "FIT504",
"PIT501", "PIT502", "PIT503"]

df = df.drop(columns=features_dropped)

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

print("Features used: ", df.columns)
print(len(df.columns))

scaler = MinMaxScaler()
scaler.fit(train_df)
joblib.dump(scaler, "scaler/lstm.gz")

train_data = scaler.transform(train_df)
test_data = scaler.transform(test_df)

# Generated training sequences for use in the model.
def create_sequences(values, history_size, target_size, step):
    data = []
    target = []

    start_index = history_size

    for i in range(start_index, len(values) -  target_size):
      indices = range(i - history_size, i, step)
      data.append(values[indices])
      target.append(values[i:i+target_size])
    
    return np.array(data), np.array(target)


x_train, y_train = create_sequences(train_data, 20, 20, 1)
x_test, y_test = create_sequences(test_data, 20, 20, 1)

print("Training input shape: ", x_train.shape)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(50000).batch(256).repeat()

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.cache().shuffle(50000).batch(256).repeat()

model = tf.keras.models.Sequential()
model.add(LSTM(units=64, input_shape=x_train.shape[1:], return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dense(units=x_train.shape[2]))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(
  train_data, 
  epochs=args.epoch,
  steps_per_epoch=300,
  validation_data=test_data,
  validation_steps=50,
  callbacks=[early_stopping]
)

loss, mean_error = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mean_error}")

model.save('model/lstm')

plot_train_history(history, "Training vs Val Loss")