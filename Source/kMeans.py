import pandas as pd
from sklearn import preprocessing
from sklearn.KMeans import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv('sonHali.csv')
le = preprocessing.LabelEncoder()

host_is_superhost=data['host_is_superhost']
neighbourhood_cleansed=data['neighbourhood_cleansed']
room_type=data['room_type']
accommodates=le.fit_transform(data['accommodates'])
bathrooms_text=le.fit_transform(data['bathrooms_text'])
bedrooms=le.fit_transform(data['bedrooms'])
beds=le.fit_transform(data['beds'])
features=list(zip(host_is_superhost,neighbourhood_cleansed,room_type,accommodates,bathrooms_text,bedrooms,beds))
label=data.price
km=KMeans(n_clusters=150,init='k-means++',random_state=0)
km.fit(features)
centroids=km.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.show()