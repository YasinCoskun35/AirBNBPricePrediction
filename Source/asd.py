# Regression Example With Boston Dataset: Standardized and Larger
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier 
# load dataset
data = pd.read_csv('enSonHali.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
columnsToDelete = ['minimum_nights','maximum_nights','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
                   'review_scores_communication','review_scores_location','review_scores_value','instant_bookable','calculated_host_listings_count']
data  = data.drop(columns=columnsToDelete)
# split into input (X) and output (Y) variables
X = data.iloc[:,0:7]
Y = data.iloc[:,7]
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
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2)
# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
    
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics =['accuracy'])
    
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=larger_model, epochs=10, batch_size=5)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=2)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))