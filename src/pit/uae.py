import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import util

from argparse import ArgumentParser
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
parser.add_argument("-d", "--dataset", default="dataset/swat-minimized.csv", type=str)
parser.add_argument("-ht", "--history", default=10, type=int)
parser.add_argument("-c", "--compact", default=0.5, type=float)
parser.add_argument("-f", "--filename", default="uae", type=str)

args = parser.parse_args()

print(args)
print("History size:", args.history)

df = pd.read_csv(args.dataset)

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

model = Autoencoder(x_train.shape, args.compact)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(
    train_tensor, 
    epochs=args.epoch,
    steps_per_epoch=300,
    validation_data=val_tensor,
    validation_steps=100
)

loss, mae = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mae}")

model.save(f'model/pit/{args.filename}-{history_size}')
joblib.dump(scaler, f"scaler/pit/{args.filename}-{history_size}.gz")

error_mean, error_std = util.calculate_error(model, val_data, history_size)

np.save(f"npy/pit/{args.filename}/mean-{history_size}", error_mean)
np.save(f"npy/pit/{args.filename}/std-{history_size}", error_std)

util.plot_train_history(history, "Training vs Val Loss", f"plot/pit/{args.filename}-{history_size}.png")