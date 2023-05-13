import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import util

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Reshape

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)
parser.add_argument("-d", "--dataset", default="dataset/swat-minimized.csv", type=str)
parser.add_argument("-ht", "--history", default=10, type=int)

args = parser.parse_args()

df = pd.read_csv(args.dataset)
df = df.drop("timestamp", axis=1)

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

features_considered = []
for column in df.columns:
  ks_result = stats.ks_2samp(train_df[column],test_df[column])
  print(column, ks_result)
  if (ks_result.statistic < 0.2):
    features_considered.append(column)

df = df[features_considered]
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

print("Training input shape: ", x_train.shape, y_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

val_tensor = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_tensor = val_tensor.cache().shuffle(50000).batch(256).repeat()

model = tf.keras.models.Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(units=x_train.shape[2]))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(
  train_tensor, 
  epochs=args.epoch,
  steps_per_epoch=300,
  validation_data=val_tensor,
  validation_steps=100,
  callbacks=[early_stopping]
)

loss, mean_error = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mean_error}")

error_mean, error_std = util.calculate_error(model, train_data, history_size)

np.save("npy/cnn/mean", error_mean)
np.save("npy/cnn/std", error_std)
model.save('model/cnn')
joblib.dump(scaler, "scaler/cnn.gz")

util.plot_train_history(history, "Training vs Val Loss", "plot/cnn.png")