import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy
from scipy.signal import medfilt
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product

df = pd.read_csv("dataset/swat-2015-data.csv", delimiter=";", decimal=",")
df = df[16000:]
df = df[::5]
df = df.drop("Normal/Attack", axis=1)
df = df.drop("Timestamp", axis=1)

attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

features_dropped = ["AIT201", "AIT202", "AIT203", "P201", "AIT401",
"AIT402", "AIT501", "AIT502", 'AIT503', "AIT504", "FIT503", "FIT504",
"PIT501", "PIT502", "PIT503"]

df = df.drop(columns=features_dropped)
attack_df = attack_df.drop(columns=features_dropped)

# Function for repeatability
def Random(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

def arch(param, data):
    Random(0)
    input_dots = Input((36,))

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

    out = Dense(36, activation='linear')(x)

    model = Model(input_dots, out)
    model.compile(optimizer=Adam(param[3]), loss='mae', metrics=["mse"])
    
    early_stopping = EarlyStopping(patience=3, verbose=0)
    model.fit(data, data,
                validation_split=0.2,
                epochs=40,
                batch_size=param[4],
                shuffle=True,
                callbacks=[early_stopping]
               )
    return model

# hyperparameters selection
BEST_PARAMS = [28, 14, 7, 0.005, 32]
Q = 0.99 # quantile for upper control limit (UCL) selection

predicted_outlier, predicted_cp = [], []
X_train = df

# scaler init and fitting
StSc = StandardScaler()
StSc.fit(X_train)

train_data = StSc.transform(X_train)
# model defining and fitting
model = arch(BEST_PARAMS, train_data)

# results predicting
residuals = pd.DataFrame(train_data - model.predict(train_data)).abs().sum(axis=1)
UCL = residuals.quantile(Q)

attack_data = StSc.transform(attack_df.drop("Normal/Attack", axis=1))
ae_residuals = attack_data - model.predict(attack_data)
ae = pd.DataFrame(ae_residuals).abs().sum(axis=1)

prediction = pd.Series((ae > 3/2*UCL).astype(int).values, 
                            index=attack_df.index).fillna(0)

attack_df["Anomaly"] = (ae > 3/2*UCL).astype(int).values

print(attack_df.to_csv("dataset/anomaly-test-swat.csv"))
# predicted outliers saving
# predicted_outlier.append(prediction)

# # predicted CPs saving
# prediction_cp = abs(prediction.diff())
# prediction_cp[0] = prediction[0]
# predicted_cp.append(prediction_cp)

# # true outlier indices selection
# true_outlier = [df.anomaly for df in list_of_df]

# predicted_outlier[0].plot(figsize=(12, 3), label='predictions', marker='o', markersize=5)
# true_outlier[0].plot(marker='o', markersize=2)
# plt.savefig('plot/autoencoder-swat.png')