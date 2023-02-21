import pandas as pd
import window_generator
from argparse import ArgumentParser
from autoencoder import Autoencoder
from scipy import stats
from sklearn.preprocessing import StandardScaler

parser = ArgumentParser()
parser.add_argument("-p", "--path")
parser.add_argument("-e", "--epoch", default=1, type=int)

args = parser.parse_args()

df = pd.read_csv(args.path, delimiter=";", decimal=",")
date_time = pd.to_datetime(df.pop('Timestamp'))

df = df.set_index(date_time)
df = df.drop("Normal/Attack", axis=1)

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

scaler = StandardScaler()
scaler.fit(df)
columns = df.columns

data = scaler.transform(train_df)
index = train_df.index
train_df = pd.DataFrame(data)
train_df.index = index
train_df.columns = columns

data = scaler.transform(val_df)
index = val_df.index
val_df = pd.DataFrame(data)
val_df.index = index
val_df.columns = columns

data = scaler.transform(test_df)
index = test_df.index
test_df = pd.DataFrame(data)
test_df.index = index
test_df.columns = columns

features_considered = []
for column in df.columns:
  ks_result = stats.ks_2samp(train_df[column],val_df[column])
  if (ks_result.statistic < 0.02):
    features_considered.append(column)

print("Features used: ", features_considered)

train_df = train_df[features_considered]
val_df = val_df[features_considered]
test_df = val_df[features_considered]

window = window_generator.WindowGenerator(5, 1, 1, train_df.columns, ["FIT101"])

train = window.make_dataset(train_df)
val = window.make_dataset(val_df)
test = window.make_dataset(test_df)

autoencoder = Autoencoder(train, val, epochs=args.epoch)
BEST_PARAMS = [5, 4, 2, 0.005, 32]
model = autoencoder.compile_fit(BEST_PARAMS)

# inputs, labels = next(iter(train))
# plot_col_index = window.column_indices["FIT101"]
# for n in range(len(inputs)):
#   print(inputs[n, :, plot_col_index].numpy())
#   prediction = model.predict(inputs)
#   print(prediction[n, :, plot_col_index])

result = model.evaluate(test)
for i in range (len(result)):
  print(f"{model.metrics_names[i]}: {result[i]}")

model.save('model/autoencoder')

