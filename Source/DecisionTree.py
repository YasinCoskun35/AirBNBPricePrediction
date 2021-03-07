import pandas as pd
from pandas.core.algorithms import mode
from scipy.sparse.construct import random
from sklearn import preprocessing
from sklearn import tree

data=pd.read_csv('outliersCleaned.csv')
from sklearn.model_selection import train_test_split
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
clf=tree.DecisionTreeRegressor()
clf.fit(features,target)
tree.plot_tree(clf)
predictions=clf.predict(X_test)
predictedData=[]
testData=[]
i=0
from sklearn.metrics import mean_absolute_error
validation_prediction_errors = mean_absolute_error(y_test, predictions)
print(len(predictions))
print(validation_prediction_errors)
correctPredictedData=0
for  yTestValue in y_test:
    yNew=clf.predict([X_test[i]])
    predictedData.append(yNew[0])
    testData.append(yTestValue)
    compareRate=yTestValue/yNew
    ##print("Test Value: "+str(yTestValue)+"  Actual Data:   "+str(yNew[0]))
    if yTestValue<yNew[0]+50 and yTestValue>yNew[0]-50:
        correctPredictedData+=1
    i+=1
print("correct predicted data number is : ----->>>>"+str(correctPredictedData))