import pandas as pd
from pandas.core.algorithms import mode
from scipy.sparse.construct import random
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data=pd.read_csv('sonHali.csv')

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


model = MultinomialNB()

model.fit(X_train,y_train)
i=0
predictedData=[]
testData=[]
correctPredictCount=0
asiriSapanSayisi=0
for  yTestValue in y_test:
    yNew=model.predict([X_test[i]])
    predictedData.append(yNew[0])
    testData.append(yTestValue)
    compareRate=yTestValue/yNew
    if yTestValue==yNew[0]:
        correctPredictCount+=1
    if compareRate>3 or compareRate<1/3:
        asiriSapanSayisi+=1
    i+=1

lenAfterSubstract=len(y_test)-asiriSapanSayisi

# Test datasından sapan değerleri çıkartıp ,kalan test datası ile tekrar accuracy hesapla
print("Accuracy:",metrics.accuracy_score(testData, predictedData))
print(len(testData))
print("asiri sapan deger sayisi: "+ str(asiriSapanSayisi))
print("Yeni oran  - - -- "+str((correctPredictCount/lenAfterSubstract)*100))


dataObject={'Predicted Data':predictedData,'Test Data':testData}
compareDf=pd.DataFrame(data=dataObject)
compareDf.to_csv('ComparisonExcel.csv')
#SENSİTİVİTY SPECİFİTY ACCURACY CONFUSION MATRIX



