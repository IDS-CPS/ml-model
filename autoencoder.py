import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, TimeDistributed, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class Autoencoder():
  def __init__(self, train_data, validation_data, epochs=1):
    self.train = train_data
    self.validation = validation_data
    self.epochs = epochs

  def num_targets(self):
    for inputs, labels in self.train.take(1):
      return labels.shape[2]

  def input_shape(self):
    for inputs, labels in self.train.take(1):
      return (inputs.shape[1], inputs.shape[2])

  # Function for repeatability
  def random(self, seed_value):
      # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
      os.environ['PYTHONHASHSEED']=str(seed_value)

      # 2. Set `python` built-in pseudo-random generator at a fixed value
      random.seed(seed_value)

      # 3. Set `numpy` pseudo-random generator at a fixed value
      np.random.seed(seed_value)

      # 4. Set `tensorflow` pseudo-random generator at a fixed value
      tf.random.set_seed(seed_value)

  def compile(self, param):
    self.random(0)
    input_dots = Input(self.input_shape())

    x = Dense(param[0])(input_dots)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(param[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    bottleneck = Dense(param[2], activation='linear')(x)

    x = Dense(param[1])(bottleneck)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(param[0])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(self.num_targets(), activation='linear')(x)

    model = Model(input_dots, out)
    model.compile(
      optimizer=Adam(param[3]), 
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    return model

  def fit(self, model, param):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    model.fit(
      self.train,
      validation_data=self.validation,
      epochs=self.epochs,
      callbacks=[early_stopping]
    )
    
    return model

  def compile_fit(self, param):
    model = self.compile(param)

    model = self.fit(model, param)

    return model
