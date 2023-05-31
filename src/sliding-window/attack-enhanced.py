import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve

parser = ArgumentParser()
parser.add_argument("-t", "--threshold", default=7, type=int)
parser.add_argument("-th", "--timethreshold", default=6, type=int)
parser.add_argument("-ht", "--history", default=10, type=int)
parser.add_argument("-m", "--model", type=str)

args = parser.parse_args()
history_size = args.history
threshold = args.threshold
time_threshold = args.timethreshold

scaler_src = f"scaler/{args.model}-{history_size}.gz"
model_src = f"model/{args.model}-{history_size}"
mean_npy = f"npy/uae/mean-enhanced-{history_size}.npy"
std_npy = f"npy/uae/std-enhanced-{history_size}.npy"

print(history_size, threshold, time_threshold, args.model)
print(args)

attack_df = pd.read_csv("dataset/swat-attack.csv", delimiter=";", decimal=",")
attack_df.columns = [column.strip() for column in attack_df.columns]
attack_df["Normal/Attack"] = attack_df["Normal/Attack"].replace(["A ttack"], "Attack")
attack_df = attack_df.set_index("Timestamp")
attack_df = attack_df[::5]

features_dropped = ["AIT201", "AIT202", "AIT203", "P201", "AIT401",
"AIT402", "AIT501", "AIT502", 'AIT503', "AIT504", "FIT503", "FIT504",
"PIT501", "PIT502", "PIT503", "P603"]
attack_df = attack_df.drop(columns=features_dropped)

for column in attack_df.columns:
    if column != "Normal/Attack":
        attack_df[f"{column}-past"] = attack_df[column].shift(1)

attack_df = attack_df.reset_index(drop=True)
attack_df = attack_df[1:]
scaler = joblib.load(scaler_src)
attack_data = scaler.transform(attack_df.drop("Normal/Attack", axis=1))

mean = np.load(mean_npy)
std = np.load(std_npy)
is_attack_period = False
attack_number = 0
attack_detected = set()

consecutive_counter = 0
model = tf.keras.models.load_model(model_src)
attack_df["Prediction"] = "Normal"
for i in range (len(attack_data)-history_size):
    end_index = i + history_size
    input_window = attack_data[i:end_index]
    target_window = attack_data[end_index]

    prediction = model.predict(np.expand_dims(input_window, axis=0), verbose=0).squeeze()
    error = np.abs(prediction - target_window)

    z_score_all = np.abs(error-mean)/std
    z_score_max = np.nanmax(z_score_all)
    attack_label = attack_df.iloc[end_index]['Normal/Attack']

    if (not is_attack_period and attack_label == 'Attack'):
        is_attack_period = True
        attack_number += 1

    if (is_attack_period and attack_label == 'Normal'):
        is_attack_period = False

    if z_score_max > threshold:
        consecutive_counter += 1
    else:
        consecutive_counter = 0

    if consecutive_counter > time_threshold:
        start_attack = attack_df.index[end_index-consecutive_counter]
        end_attack = attack_df.index[end_index]
        attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"

        if attack_label == 'Attack':
            attack_detected.add(attack_number)

if consecutive_counter > time_threshold:
    start_attack = attack_df.index[len(attack_data)-consecutive_counter-1]
    end_attack = attack_df.index[len(attack_data)-1]
    attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"


attack_df = attack_df[history_size+1:]
real_value = attack_df["Normal/Attack"].to_numpy()
real_value[real_value == "Normal"] = 0
real_value[(real_value == "Attack")] = 1

predicted_value = attack_df["Prediction"].to_numpy()
predicted_value[predicted_value == "Normal"] = 0
predicted_value[predicted_value == "Attack"] = 1

real_value = np.array(real_value, dtype=int)
predicted_value = np.array(predicted_value, dtype=int)

print(f"{len(attack_detected)} out of {attack_number} detected")
print(attack_detected)

print("Precision:", precision_score(real_value, predicted_value))
print("Recall:", recall_score(real_value, predicted_value))
print("F1 Score:", f1_score(real_value, predicted_value))
fpr, tpr, thresholds = roc_curve(real_value, predicted_value)
print("AUC:", auc(fpr, tpr))


attack_df[["Normal/Attack", "Prediction"]].to_csv(f"dataset/result/{args.model}-{history_size}-{threshold}-{time_threshold}.csv")