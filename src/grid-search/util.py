import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

def create_sequences(values, history_size):
    data = []
    target = []

    for i in range(len(values)-history_size):
        end_index = i + history_size
        data.append(values[i:end_index])
        target.append(values[end_index])
    
    return np.array(data), np.array(target)

def calculate_error(model, data, history_size):
    error_arr = []
    for i in range (len(data)-history_size):
        end_index = i + history_size
        input_window = data[i:end_index]
        target_window = data[end_index]

        prediction = model.predict(np.expand_dims(input_window, axis=0), verbose=0).squeeze()
        error = np.abs(prediction - target_window)

        error_arr.append(error)

    error_arr = np.asarray(error_arr)

    error_mean = np.mean(error_arr, axis=0)
    error_std = np.std(error_arr, axis=0)

    return error_mean, error_std
