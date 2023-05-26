import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import util

from itertools import product
from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)
parser.add_argument("-d", "--dataset", default="dataset/pompa-train.csv", type=str)
parser.add_argument("-ht", "--history", default=10, type=int)

args = parser.parse_args()

df = pd.read_csv(args.dataset)

n = len(df)

features_considered = ['adc_level', 'adc_flow', 'adc_pressure_left', 'adc_pressure_right', 'level', 'flow', 'pressure_left', 'pressure_right']

df = df[features_considered]
train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):]

print("Features used: ", df.columns)
print(len(df.columns))

scaler = MinMaxScaler()
scaler.fit(train_df)

train_data = scaler.transform(train_df)
val_data = scaler.transform(val_df)

history_size = args.history

x_train, y_train = util.create_sequences(train_data, history_size)
x_val, y_val = util.create_sequences(val_data, history_size)

print("Training input shape: ", x_train.shape, y_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

def create_model(n_filter=32, dropout_rate=0.5, kernel_size=2, pool_size=2):
  model = tf.keras.models.Sequential()
  model.add(Conv1D(filters=n_filter, kernel_size=kernel_size, activation='relu', input_shape=x_train.shape[1:]))
  model.add(MaxPooling1D(pool_size=pool_size, strides=1))
  model.add(Conv1D(filters=n_filter*2, kernel_size=kernel_size, activation='relu'))
  model.add(MaxPooling1D(pool_size=pool_size, strides=1))
  model.add(Conv1D(filters=n_filter*4, kernel_size=kernel_size, activation='relu'))
  model.add(MaxPooling1D(pool_size=pool_size, strides=1))
  model.add(Conv1D(filters=n_filter*8, kernel_size=kernel_size, activation='relu'))
  model.add(MaxPooling1D(pool_size=pool_size, strides=1))
  model.add(Flatten())
  model.add(Dropout(rate=dropout_rate))
  model.add(Dense(units=x_train.shape[2]))

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  return model


n_filters = [16, 32, 64]
dropout_rate = [0.1, 0.3, 0.5, 0.7]
kernel_size = [2]
pool_size = [2]

params = list(product(n_filters, dropout_rate, kernel_size, pool_size))

grid_results = []
for param in params:
  model = create_model(n_filter=param[0], dropout_rate=param[1], kernel_size=param[2], pool_size=[3])

  model.summary()
  history = model.fit(
    train_tensor, 
    epochs=args.epoch,
    steps_per_epoch=100,
    verbose=False
  )

  train_loss, train_mae = model.evaluate(x_train, y_train, verbose=False)
  loss, mean_error = model.evaluate(x_val, y_val, verbose=False)

  grid_results.append({
    "n_filter": param[0],
    "dropout_rate": param[1],
    "kernel_size": param[2],
    "pool_size": param[3],
    "loss": loss,
    "mae": mean_error,
    "train_loss": train_loss,
    "train_mae": train_mae
  })

df = pd.DataFrame.from_dict(grid_results)
df = df.sort_values(by=["mae", "loss"])

print(df.to_string())