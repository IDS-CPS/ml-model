import pandas as pd
import window_generator
import tensorflow as tf
import logging

from argparse import ArgumentParser
from scipy import stats
from sklearn.preprocessing import StandardScaler

parser = ArgumentParser()
parser.add_argument("-p", "--path")
parser.add_argument("-e", "--epoch", default=1, type=int)

logging.basicConfig(filename="logs/cnn_2/cnn.log", level=logging.INFO)

args = parser.parse_args()

df = pd.read_csv(args.path, delimiter=";", decimal=",")
date_time = pd.to_datetime(df.pop('Timestamp'))

df = df.set_index(date_time)
df = df.drop("Normal/Attack", axis=1)

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

scaler = StandardScaler()
scaler.fit(df)
columns = df.columns

data = scaler.transform(train_df)
index = train_df.index
train_df = pd.DataFrame(data)
train_df.index = index
train_df.columns = columns

data = scaler.transform(val_df)
index = val_df.index
val_df = pd.DataFrame(data)
val_df.index = index
val_df.columns = columns

data = scaler.transform(test_df)
index = test_df.index
test_df = pd.DataFrame(data)
test_df.index = index
test_df.columns = columns

features_considered = []
for column in df.columns:
  ks_result = stats.ks_2samp(train_df[column],val_df[column])
  if (ks_result.statistic < 0.02):
    features_considered.append(column)

print("Features used: ", features_considered)

train_df = train_df[features_considered]
val_df = val_df[features_considered]
test_df = val_df[features_considered]

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(5,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

for feature in features_considered:
  window = window_generator.WindowGenerator(24, 1, 24, train_df.columns, [feature])

  train = window.make_dataset(train_df)
  val = window.make_dataset(val_df)
  test = window.make_dataset(test_df)

  # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
  #                                                     patience=10,
  #                                                     mode='min')

  csv_logger = tf.keras.callbacks.CSVLogger(f"logs/cnn_2/{feature}.log")

  conv_model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = conv_model.fit(train, epochs=args.epoch,
                      validation_data=val,
                      callbacks=[csv_logger])

  evaluation = conv_model.evaluate(test)
  logging.info(f"{feature}: {dict(zip(conv_model.metrics_names, evaluation))}")

  conv_model.save(f'model/cnn_2/{feature}.h5')