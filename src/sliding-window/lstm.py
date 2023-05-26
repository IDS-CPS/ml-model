import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import util

from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)
parser.add_argument("-d", "--dataset", default="dataset/swat-minimized.csv", type=str)
parser.add_argument("-ht", "--history", default=10, type=int)
parser.add_argument("-n", "--units", default=100, type=int)

args = parser.parse_args()

df = pd.read_csv(args.dataset, delimiter=";", decimal=",")
df = df[16000:]
df = df[::5]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)


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

train_data = scaler.transform(train_df)
test_data = scaler.transform(test_df)

history_size = args.history

x_train, y_train = util.create_sequences(train_data, history_size)
x_test, y_test = util.create_sequences(test_data, history_size)

print("Training input shape: ", x_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

val_tensor = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_tensor = val_tensor.cache().shuffle(50000).batch(256).repeat()

n_units = args.units
model = tf.keras.models.Sequential([ 
    LSTM(n_units, return_sequences=True, input_shape=x_train.shape[1:]),
    LSTM(n_units, return_sequences=True),
    LSTM(n_units),
    Dense(x_train.shape[2]),
]) 

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(
  train_tensor, 
  epochs=args.epoch,
  steps_per_epoch=100,
  validation_data=val_tensor,
  validation_steps=50,
  callbacks=[early_stopping]
)

loss, mean_error = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mean_error}")

model.save(f'model/lstm-new-{history_size}')
joblib.dump(scaler, f"scaler/lstm-new-{history_size}.gz")

error_mean, error_std = util.calculate_error(model, train_data, history_size)

np.save(f"npy/lstm-new/mean-{history_size}", error_mean)
np.save(f"npy/lstm-new/std-{history_size}", error_std)

util.plot_train_history(history, "Training vs Val Loss", f"plot/lstm-new-{history_size}.png")