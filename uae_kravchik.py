import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Model, layers

class Encoder(layers.Layer):
  def __init__(self, intermediate_dim=32):
    super(Encoder, self).__init__()
    self.flatten = layers.Flatten()
    self.hidden_layer = layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu
    )
    self.output_layer = layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu
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
      activation=tf.nn.relu
    )
    self.output_layer = layers.Dense(
      units=original_dim,
      activation=tf.nn.relu
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

train_data = scaler.transform(train_df)
test_data = scaler.transform(test_df)

mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)

np.save('uae_mean.npy', mean)
np.save('uae_std.npy', std)

TIME_STEPS = 40
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(train_data)
x_test = create_sequences(test_data)

print("Training input shape: ", x_train.shape)

model = Autoencoder(x_train.shape)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.experimental.AdamW(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

model.fit(
  x_train, 
  x_train,
  epochs=args.epoch,
  batch_size=32,
  validation_split=0.1,
  callbacks=[early_stopping]
)

loss, mae = model.evaluate(x_test, x_test)

print(f"Loss: {loss}, Mean Absolute Error: {mae}")

model.save('model/uae-kravchik')