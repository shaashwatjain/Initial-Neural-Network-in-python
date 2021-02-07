import pandas as pd
import numpy as np

data = pd.read_csv('../LBW_Dataset.csv')        #Path to source dataset to be cleaned
df = pd.DataFrame(data)

np.random.seed(0)

# normalizing dataframe columns 
def normalize(df):
	result = df.copy()
	for feature_name in df.columns:
		max_value = df[feature_name].max()
		min_value = df[feature_name].min()
		if min_value == max_value:
			result[feature_name] = df[feature_name] / max_value
		else:
			result[feature_name] = (
					df[feature_name] - min_value) / (max_value - min_value)
	return result

# dropping rows that contain outliers
for feature in df.columns:
    deviation = df[feature].std()
    mean = df[feature].mean()
    tolerance = 2
    to_drop = []
    for i,value in enumerate(df[feature]):
        if value > mean + (tolerance*deviation) or value < mean + (tolerance*deviation):
            to_drop.append(i)
    for row in to_drop:
        df.drop(df.index[row])

# replacing values with the mode of the columns 
for feature in df.columns:
    if df[feature].isnull().any():
        mode = df[feature].mode()
        mode = float(mode)
        df[feature].fillna(mode, inplace = True)

new_df = normalize(df)
df = new_df

# shuffling to prevent sampling errors 
df = df.sample(frac = 1)

df.to_csv("../data/cleaned_Dataset.csv")