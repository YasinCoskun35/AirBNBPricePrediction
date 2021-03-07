import pandas as pd 
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
data=pd.read_csv('outliersCleaned.csv')

print(data)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
data.to_csv("normalizedData.csv")
print(scaled)