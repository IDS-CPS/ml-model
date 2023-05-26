import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import util

from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense

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
test_data = scaler.transform(val_df)

history_size = args.history

x_train, y_train = util.create_sequences(train_data, history_size)
x_test, y_test = util.create_sequences(test_data, history_size)

print("Training input shape: ", x_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

def create_model(n_units=32):
  model = tf.keras.models.Sequential()
  model.add(LSTM(n_units, return_sequences=True, input_shape=x_train.shape[1:]))
  model.add(LSTM(n_units, return_sequences=True))
  model.add(LSTM(n_units))
  model.add(Dense(units=x_train.shape[2]))

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  return model

n_units = [16, 32, 64, 100]
grid_results = []

for unit in n_units:
  model = create_model(unit)
  history = model.fit(
    train_tensor, 
    epochs=args.epoch,
    steps_per_epoch=100,
    verbose=False
  )

  train_loss, train_mae = model.evaluate(x_train, y_train, verbose=False)
  loss, mean_error = model.evaluate(x_test, y_test, verbose=False)

  grid_results.append({
    "n_units": unit,
    "loss": loss,
    "mae": mean_error,
    "train_loss": train_loss,
    "train_mae": train_mae
  })

df = pd.DataFrame.from_dict(grid_results)
df = df.sort_values(by=["mae", "loss"])

print(df.to_string())  