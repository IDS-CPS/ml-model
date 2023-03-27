import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

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

  plt.savefig("plot/uae-v4.png")

class Encoder(layers.Layer):
  def __init__(self, intermediate_dim=32):
    super(Encoder, self).__init__()
    self.flatten = layers.Flatten()
    self.hidden_layer = layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.tanh
    )
    self.output_layer = layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.tanh
    )
    
  def call(self, input_features):
    flatten = self.flatten(input_features)
    activation = self.hidden_layer(flatten)
    return self.output_layer(activation)

class Decoder(layers.Layer):
  def __init__(self, original_dim, intermediate_dim=32):
    super(Decoder, self).__init__()
    self.hidden_layer = layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.tanh
    )
    self.output_layer = layers.Dense(
      units=original_dim,
      activation=tf.nn.tanh
    )
  
  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)

class Autoencoder(tf.keras.Model):
  def __init__(self, train_shape):
    super(Autoencoder, self).__init__()
    self.train_shape = train_shape
    self.encoder = Encoder(intermediate_dim=0.5*train_shape[1]*train_shape[2])
    self.decoder = Decoder(original_dim=train_shape[1]*train_shape[2], intermediate_dim=0.5*train_shape[1]*train_shape[2])
  
  def call(self, input_features):
    code = self.encoder(input_features)
    decoded = self.decoder(code)
    output = tf.reshape(decoded, [-1, self.train_shape[1], self.train_shape[2]])
    return output

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)

args = parser.parse_args()

df = pd.read_csv("dataset/swat-2015-data.csv", delimiter=";", decimal=",")
df = df[16000:]
df = df[::5]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)


# features_considered = []
# for column in df.columns:
#   ks_result = stats.ks_2samp(train_df[column],test_df[column])
#   if (ks_result.statistic < 0.02):
#     features_considered.append(column)
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
joblib.dump(scaler, "scaler/uae-v3.gz")
columns = train_df.columns

train_data = scaler.transform(train_df)
val_data = scaler.transform(test_df)

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


x_train, y_train = create_sequences(train_data, 10, 10, 1)
x_test, y_test = create_sequences(val_data, 10, 10, 1)

print("Training input shape: ", x_train.shape)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(50000).batch(256).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_data = val_data.cache().shuffle(50000).batch(256).repeat()

model = Autoencoder(x_train.shape)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(
  train_data, 
  epochs=args.epoch,
  steps_per_epoch=100,
  validation_data=val_data,
  validation_steps=50,
  callbacks=[early_stopping]
)

loss, mae = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mae}")

model.save('model/uae-kravchik-v4')

plot_train_history(history, "Training vs Val Loss")