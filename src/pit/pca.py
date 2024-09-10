import pandas as pd
import numpy as np
import joblib

from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", default="dataset/swat-2015-data.csv", type=str)
parser.add_argument("-w", "--window_size", default=10, type=int)
parser.add_argument("-t", "--threshold", default=7, type=int)
parser.add_argument("-th", "--timethreshold", default=6, type=int)
parser.add_argument("-n", "--n_units", type=int)

args = parser.parse_args()

print(args)

df = pd.read_csv(args.dataset)
n = len(df)

train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):]

scaler = MinMaxScaler()
scaler = scaler.fit(train_df)

train_data = scaler.transform(train_df)
val_data = scaler.transform(val_df)

# Fit PCA to training data
pca = PCA(n_components=args.n_units)
pca.fit(train_data)
val_pca = pca.transform(val_data)
val_pca = pca.inverse_transform(val_pca)
error = np.abs(val_pca - val_data)

e_mean = np.mean(error, axis=0)
e_std = np.std(error, axis=0)

attack_df = pd.read_csv("dataset/pompa-attack-3.csv")
attack_df = attack_df[["adc_level","adc_temp","adc_flow","adc_pressure_left","adc_pressure_right","Class"]]

attack_data = scaler.transform(attack_df.drop("Class", axis=1))

window_size = args.window
print(window_size)

threshold = args.threshold
time_threshold = args.timethreshold
is_attack_period = False
attack_number = 0
attack_detected = set()

consecutive_counter = 0
attack_df["Prediction"] = "Normal"

np.seterr(all="ignore")
for i in range(len(attack_df)//window_size):
    start_index = window_size * i
    window = attack_data[start_index:start_index+window_size]

    window_pca = pca.transform(window)
    window_recon = pca.inverse_transform(window_pca)
    error = np.abs(window - window_recon)
    z_score_all = np.abs(error - e_mean)/e_std
    z_score_max = np.nanmax(z_score_all, axis=1)
    
    attack_label = attack_df.iloc[start_index:start_index+window_size]['Class'].values

    for j in range(len(z_score_max)):
        if (not is_attack_period and attack_label[j] == 'Attack'):
            is_attack_period = True
            start_period = attack_df.index[start_index+j]
            attack_number += 1

        if (is_attack_period and attack_label[j] == 'Normal'):
            end_period = attack_df.index[start_index+j-1]
            is_attack_period = False

        if z_score_max[j] > threshold:
            consecutive_counter += 1
        else:
            consecutive_counter = 0

        if consecutive_counter > time_threshold:
            start_attack = attack_df.index[start_index+j-consecutive_counter]
            end_attack = attack_df.index[start_index+j]
            attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"

            if 'Attack' in attack_label:
                attack_detected.add(attack_number)

            consecutive_counter = 0

if consecutive_counter > time_threshold:
    start_attack = attack_df.index[len(attack_data)-consecutive_counter-1]
    end_attack = attack_df.index[len(attack_data)-1]
    attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"

if consecutive_counter > time_threshold:
    start_attack = attack_df.index[len(attack_data)-consecutive_counter-1]
    end_attack = attack_df.index[len(attack_data)-1]
    attack_df.loc[start_attack:end_attack, "Prediction"] = "Attack"

attack_df = attack_df[window_size:]
real_value = attack_df["Class"].to_numpy()
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

joblib.dump(scaler, f"scaler/pca-{args.window_size}.gz")
np.save(f"npy/pca/mean-{args.window_size}", e_mean)
np.save(f"npy/pca/std-{args.window_size}", e_std)
joblib.dump(pca, f"model/pca-{args.window_size}.gz")

attack_df[["Normal/Attack", "Prediction"]].to_csv(f"dataset/result/pca-{window_size}-{threshold}-{time_threshold}.csv", index=False)
