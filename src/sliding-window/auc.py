import pandas as pd
import numpy as np

from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-ht", "--history", default=10, type=int)

args = parser.parse_args()

df = pd.read_csv(args.dataset)

df = df[args.history+1:]

real_value = df["Normal/Attack"].to_numpy()
predicted_value = df["Prediction"].to_numpy()

real_value = np.array(real_value, dtype=int)
predicted_value = np.array(predicted_value, dtype=int)

fpr, tpr, thresholds = roc_curve(real_value, predicted_value)
auc_score = auc(fpr, tpr)

print(auc_score)