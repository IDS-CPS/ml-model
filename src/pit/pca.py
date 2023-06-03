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
scaler = scaler.fit(df)

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

joblib.dump(scaler, f"scaler/pca-{args.window_size}-{args.n_units}.gz")
np.save(f"npy/pca/mean-{args.window_size}-{args.n_units}", e_mean)
np.save(f"npy/pca/std-{args.window_size}-{args.n_units}", e_std)
joblib.dump(pca, f"model/pca-{args.window_size}-{args.n_units}")
