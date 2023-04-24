import pandas as pd
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
def plot_train_history(history, title, filename):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.savefig(filename)

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
    self.target_output = layers.Dense(train_shape[2])

  def call(self, input_features):
    code = self.encoder(input_features)
    decoded = self.decoder(code)
    return self.target_output(decoded)

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
columns = train_df.columns

train_data = scaler.transform(train_df)
val_data = scaler.transform(test_df)

def create_sequences(values, history_size):
    data = []
    target = []

    for i in range(len(values)-history_size):
        end_index = i + history_size
        data.append(values[i:end_index])
        target.append(values[end_index])
    
    return np.array(data), np.array(target)


history_size = 10

x_train, y_train = create_sequences(train_data, history_size)
x_test, y_test = create_sequences(val_data, history_size)

print("Training input shape: ", x_train.shape, y_train.shape)

train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_tensor = train_tensor.cache().shuffle(50000).batch(256).repeat()

val_tensor = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_tensor = val_tensor.cache().shuffle(50000).batch(256).repeat()

model = Autoencoder(x_train.shape)

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

loss, mae = model.evaluate(x_test, y_test)

print(f"Loss: {loss}, Mean Absolute Error: {mae}")


error_arr = []
for i in range (len(train_data)-history_size):
    end_index = i + history_size
    input_window = train_data[i:end_index]
    target_window = train_data[end_index]

    prediction = model.predict(np.expand_dims(input_window, axis=0)).squeeze()
    error = np.abs(prediction - target_window)

    error_arr.append(error)

error_arr = np.asarray(error_arr)
error_arr = error_arr.reshape((-1, error_arr.shape[-1]))

error_mean = np.mean(error_arr, axis=0)
error_std = np.std(error_arr, axis=0)

joblib.dump(scaler, "scaler/uae.gz")
np.save("npy/uae/mean", error_mean)
np.save("npy/uae/std", error_mean)
model.save('model/uae')

plot_train_history(history, "Training vs Val Loss", "plot/uae.png")