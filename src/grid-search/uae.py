import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import util

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

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
  def __init__(self, train_shape, compact_ratio):
    super(Autoencoder, self).__init__()
    self.train_shape = train_shape
    self.encoder = Encoder(intermediate_dim=compact_ratio*train_shape[1]*train_shape[2])
    self.decoder = Decoder(original_dim=train_shape[1]*train_shape[2], intermediate_dim=compact_ratio*train_shape[1]*train_shape[2])
    self.target_output = layers.Dense(train_shape[2])

  def call(self, input_features):
    code = self.encoder(input_features)
    decoded = self.decoder(code)
    return self.target_output(decoded)

parser = ArgumentParser()
parser.add_argument("-e", "--epoch", default=1, type=int)
parser.add_argument("-d", "--dataset", default="dataset/swat-p1.csv", type=str)
parser.add_argument("-ht", "--history", default=10, type=int)

args = parser.parse_args()

print("History size:", args.history)

df = pd.read_csv(args.dataset)
df = df[16000:]
df.columns = [column.strip() for column in df.columns]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)
df = df[::5]

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

print("Features used: ", df.columns)
print(len(df.columns))

scaler = MinMaxScaler()
scaler.fit(train_df)
columns = train_df.columns

train_data = scaler.transform(train_df)
val_data = scaler.transform(test_df)

history_size = args.history

x_train, y_train = util.create_sequences(train_data, history_size)
x_test, y_test = util.create_sequences(val_data, history_size)

print("Training input shape: ", x_train.shape, y_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

val_tensor = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_tensor = val_tensor.cache().shuffle(50000).batch(256).repeat()

compact_ratio = [0.2, 0.4, 0.5, 0.7, 0.9, 1]

grid_results = []
for ratio in compact_ratio:
  model = Autoencoder(x_train.shape, ratio)
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.experimental.AdamW(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
  history = model.fit(
    train_tensor, 
    epochs=args.epoch,
    steps_per_epoch=100,
  )

  loss, mean_error = model.evaluate(x_test, y_test)

  grid_results.append({
    "compact_ratio": ratio,
    "loss": loss,
    "mae": mean_error
  })

df = pd.DataFrame.from_dict(grid_results)
df = df.sort_values(by=["mae", "loss"])

print(df.to_string())