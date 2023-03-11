import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Reshape

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)

args = parser.parse_args()

df = pd.read_csv("dataset/swat-2015-data.csv", delimiter=";", decimal=",")

df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

features_considered = []
for column in df.columns:
  ks_result = stats.ks_2samp(train_df[column],test_df[column])
  if (ks_result.statistic < 0.02):
    features_considered.append(column)

print("Features used: ", features_considered)
print(len(features_considered))

train_df = train_df[features_considered]
test_df = test_df[features_considered]

scaler = MinMaxScaler()
scaler.fit(train_df)
columns = train_df.columns

data = scaler.transform(train_df)
train_df = pd.DataFrame(data)
train_df.columns = columns

data = scaler.transform(test_df)
test_df = pd.DataFrame(data)
test_df.columns = columns

joblib.dump(scaler, 'scaler/1d-cnn.gz')

TIME_STEPS = 50
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    label = []

    for i in range(len(values) - (2 * time_steps) + 1):
        output.append(values[i : (i + time_steps)])
        label.append(values[(i + time_steps) : (i + 2*time_steps)])

    return np.stack(output), np.stack(label)

x_train, y_train = create_sequences(train_df.values)
x_test, y_test = create_sequences(test_df.values)

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

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

model.fit(
  x_train, 
  y_train,
  epochs=args.epoch,
  batch_size=32,
  validation_split=0.1,
  callbacks=[early_stopping]
)

loss, mean_error = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mean_error}")

model.save('model/1d-cnn-minmax-scaler')