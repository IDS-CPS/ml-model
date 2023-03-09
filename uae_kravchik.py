import pandas as pd
import numpy as np
import window_generator
import tensorflow as tf

from argparse import ArgumentParser
from autoencoder import Autoencoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
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
date_time = pd.to_datetime(df.pop('Timestamp'))

df = df.set_index(date_time)
df = df.drop("Normal/Attack", axis=1)

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

scaler = StandardScaler()
scaler.fit(train_df)
columns = df.columns

data = scaler.transform(train_df)
index = train_df.index
train_df = pd.DataFrame(data)
train_df.index = index
train_df.columns = columns

data = scaler.transform(test_df)
index = test_df.index
test_df = pd.DataFrame(data)
test_df.index = index
test_df.columns = columns

features_considered = []
for column in df.columns:
  ks_result = stats.ks_2samp(train_df[column],test_df[column])
  if (ks_result.statistic < 0.02):
    features_considered.append(column)

print("Features used: ", features_considered)

TIME_STEPS = 24
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(train_df.values)
x_test = create_sequences(test_df.values)

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

loss, acc = model.evaluate(x_test, x_test)

print(f"Loss: {loss}, Accuracy: {acc}")

model.save('model/autoencoder-kravchik-v2')