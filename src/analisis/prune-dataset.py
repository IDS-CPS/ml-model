import pandas as pd

df = pd.read_csv("dataset/pompa.csv")

df = df[900:27000]

df.to_csv("dataset/pompa-v2.csv", index=False)