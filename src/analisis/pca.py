import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv("dataset/train/pompa-train-v3.csv")

n = len(df)

train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):]

scaler = MinMaxScaler()
scaler = scaler.fit(train_df)

train_data = scaler.transform(train_df)
val_data = scaler.transform(val_df)

pca = PCA(n_components=5)
pca.fit(train_data)
print(pca.explained_variance_ratio_)
print(0.39809659+0.28576163+0.13647928+0.11580756)
