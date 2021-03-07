import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
data=pd.read_csv('normalizedData.csv')
le = preprocessing.LabelEncoder()
host_is_superhost=le.fit_transform(data['host_is_superhost'])
neighbourhood_cleansed=le.fit_transform(data['neighbourhood_cleansed'])
room_type=le.fit_transform(data['room_type'])
accommodates=data['accommodates']
bathrooms_text=data['bathrooms_text']
bedrooms=data['bedrooms']
beds=data['beds']
features=list(zip(host_is_superhost,neighbourhood_cleansed,room_type,accommodates,bathrooms_text,bedrooms,beds))
target=data.price
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.3)
trainData=list(zip(X_train,y_train))
testData=list(zip(X_test,y_test))

compareDf=pd.DataFrame(data=trainData)
compareDf.to_csv("trainDataNorm.csv")
comparetDf=pd.DataFrame(data=testData)
comparetDf.to_csv("testDataNorm.csv")