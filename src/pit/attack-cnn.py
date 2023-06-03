import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from itertools import product
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", type=str)

args = parser.parse_args()
history_opt = [20, 40, 80]
threshold_opt = [8, 9, 10]
time_threshold_opt = [5, 6]

params = list(product(history_opt, threshold_opt, time_threshold_opt))

df = pd.read_csv(args.dataset)
df = df[["adc_level","adc_temp","adc_flow","adc_pressure_left","adc_pressure_right","Class"]]

grid_results = []
for param in params:
    history_size = param[0]
    threshold = param[1]
    time_threshold = param[2]

    scaler_src = f"scaler/pit/cnn-{history_size}.gz"
    model_src = f"model/pit/cnn-{history_size}"
    mean_npy = f"npy/pit/cnn/mean-{history_size}.npy"
    std_npy = f"npy/pit/cnn/std-{history_size}.npy"

    model = tf.keras.models.load_model(model_src)
    mean = np.load(mean_npy)
    std = np.load(std_npy)

    scaler = joblib.load(scaler_src)
    attack_data = scaler.transform(df.drop("Class", axis=1))

    consecutive_counter = 0
    prediction_arr = []
    df["Prediction"] = "Normal"

    is_attack_period = False
    attack_number = 0
    attack_detected = set()
    for i in range (len(attack_data)-history_size):
        end_index = i + history_size
        input_window = attack_data[i:end_index]
        target_window = attack_data[end_index]

        prediction = model.predict(np.expand_dims(input_window, axis=0), verbose=0).squeeze()
        error = np.abs(prediction - target_window)

        prediction_arr.append(prediction)

        z_score_all = np.abs(error-mean)/std
        z_score_max = np.nanmax(z_score_all)
        attack_label = df.iloc[end_index]['Class']

        if (not is_attack_period and attack_label == 'Attack'):
            is_attack_period = True
            attack_number += 1

        if (is_attack_period and attack_label == 'Normal'):
            is_attack_period = False

        # print(z_score_max, attack_label)
        if z_score_max > threshold:
            consecutive_counter += 1
        else:
            consecutive_counter = 0

        if consecutive_counter > time_threshold:
            start_attack = df.index[end_index-consecutive_counter]
            end_attack = df.index[end_index]
            df.loc[start_attack:end_attack, "Prediction"] = "Attack"

            if attack_label == 'Attack':
                attack_detected.add(attack_number)

    if consecutive_counter > time_threshold:
        start_attack = df.index[len(attack_data)-consecutive_counter-1]
        end_attack = df.index[len(attack_data)-1]
        df.loc[start_attack:end_attack, "Prediction"] = "Attack"

    attack_df = df[history_size+1:]

    real_value = attack_df["Class"].to_numpy()
    real_value[real_value == "Normal"] = 0
    real_value[(real_value == "Attack")] = 1

    predicted_value = attack_df["Prediction"].to_numpy()
    predicted_value[predicted_value == "Normal"] = 0
    predicted_value[predicted_value == "Attack"] = 1

    real_value = np.array(real_value, dtype=int)
    predicted_value = np.array(predicted_value, dtype=int)

    print(param, attack_detected)

    fpr, tpr, thresholds = roc_curve(real_value, predicted_value)

    grid_results.append({
        "attack_detected": len(attack_detected),
        "precision": precision_score(real_value, predicted_value),
        "recall": recall_score(real_value, predicted_value),
        "f1_score": f1_score(real_value, predicted_value),
        "auc": auc(fpr, tpr)
    })

    attack_df.to_csv(f"dataset/pit/cnn-{history_size}-{threshold}-{time_threshold}.csv")

    df = df.drop("Prediction", axis=1)

df = pd.DataFrame.from_dict(grid_results)
df = df.sort_values(by=["attack_detected", "f1_score"])

print("Total attacks:", attack_number)
print(df.to_string())