import tensorflow 
import pandas as pd 

from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = pd.read_csv('outliersCleaned.csv')

print("tanrım kötü kullarını sen affetsen ben affetmem")