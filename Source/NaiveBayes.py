import pandas as pd
from pandas.core.algorithms import mode
from scipy.sparse.construct import random
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
data=pd.read_csv('sonHali.csv')

from sklearn.model_selection import train_test_split
le = preprocessing.LabelEncoder()

host_is_superhost=data['host_is_superhost']
neighbourhood_cleansed=data['neighbourhood_cleansed']
room_type=data['room_type']
accommodates=le.fit_transform(data['accommodates'])
bathrooms_text=le.fit_transform(data['bathrooms_text'])
bedrooms=le.fit_transform(data['bedrooms'])
beds=le.fit_transform(data['beds'])
features=list(zip(host_is_superhost,neighbourhood_cleansed,room_type,accommodates,bathrooms_text,bedrooms,beds))
target=data.price

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2)


model = MultinomialNB()

model.fit(X_train,y_train)



xNew=[0,11,1,3,1,5,3]
print(xNew)
yNew=model.predict([xNew])
print(yNew)



#SENSİTİVİTY SPECİFİTY ACCURACY CONFUSION MATRIX