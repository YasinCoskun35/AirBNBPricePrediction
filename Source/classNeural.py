import pandas
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
# load dataset
data = pandas.read_csv("enSonHali.csv")
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
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.3)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
 
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=7, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))