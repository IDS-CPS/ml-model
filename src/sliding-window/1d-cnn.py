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
parser.add_argument("-h", "--history", default=10, type=int)

args = parser.parse_args()

df = pd.read_csv(args.dataset, delimiter=";", decimal=",")
df = df[16000:]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)

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
joblib.dump(scaler, "scaler/1d-cnn.gz")

train_data = scaler.transform(train_df)
test_data = scaler.transform(test_df)

def create_sequences(values, history_size, target_size):
    data = []
    target = []

    for i in range(len(values)//history_size-1):
        start_index = history_size * i
        end_index = start_index + history_size
        data.append(values[start_index:end_index])
        target.append(values[end_index:end_index+target_size])
    
    return np.array(data), np.array(target)

history_size = args.history

x_train, y_train = create_sequences(train_data, history_size, target_size)
x_test, y_test = create_sequences(test_data, history_size, target_size)

print("Training input shape: ", x_train.shape, y_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

val_tensor = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_tensor = val_tensor.cache().shuffle(50000).batch(256).repeat()


model = tf.keras.models.Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(units=y_train.shape[1] * y_train.shape[2]))
model.add(Reshape((-1, y_train.shape[1], y_train.shape[2])))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)

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

model.save('model/1d-cnn')

error_arr = []
for i in range (len(test_data)//history_size-1):
    start_index = history_size * i
    end_index = start_index + history_size
    input_window = test_data[start_index:start_index+history_size]
    target_window = test_data[end_index:end_index+target_size]

    prediction = model.predict(input_window.reshape(1, history_size, -1)).reshape((target_window.shape[0], target_window.shape[1]))
    error = np.abs(prediction - target_window)

    error_arr.append(error)

error_arr = np.asarray(error_arr)
error_arr = error_arr.reshape((-1, error_arr.shape[-1]))

error_mean = np.mean(error_arr, axis=0)
error_std = np.std(error_arr, axis=0)

np.save("npy/cnn/mean.py", error_mean)
np.save("npy/cnn/std.py", error_mean)

plot_train_history(history, "Training vs Val Loss", "plot/cnn.png")