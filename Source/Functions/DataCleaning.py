import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
def checkColumnUsability(data):
    removedMissingData=checkMissingValues(data)
    finalData=dataAdapt(removedMissingData)
    return finalData

def checkMissingValues(data):
    for columnName in data.columns:
        if data[columnName].isna().sum()>data.shape[0]*0.8:
            data=data.drop(columnName)
    
    saveDataChange(data)
    return data

def dataAdapt(data):
    data['price']=data['price'].map(lambda x: x.lstrip('$').replace(',',''))
    data['price'] = pd.to_numeric(data['price'])
    data['host_response_rate'] = data['host_response_rate'].str.rstrip('%').astype('float')
    data['host_acceptance_rate'] = data['host_acceptance_rate'].str.rstrip('%').astype('float')
    data['host_is_superhost'].replace('t',1,inplace=True)
    data['host_is_superhost'].replace('f',0,inplace=True)
    data['instant_bookable'].replace('t',1,inplace=True)
    data['instant_bookable'].replace('f',0,inplace=True)
    data['host_has_profile_pic'].replace('t',1,inplace=True)
    data['host_has_profile_pic'].replace('f',0,inplace=True)
    data['host_identity_verified'].replace('t',1,inplace=True)
    data['host_identity_verified'].replace('f',0,inplace=True)
    data['has_availability'].replace('t',1,inplace=True)
    data['has_availability'].replace('f',0,inplace=True)
    data['bathrooms_text'].fillna('1',inplace=True)
    data['bathrooms_text']=data['bathrooms_text'].map(lambda x: str(x).rstrip('bath private shared'))
    data['bathrooms_text'].replace('Half-',1,inplace=True)
    data['bathrooms_text'].replace('Shared half-',1,inplace=True)
    data['bathrooms_text'].replace('Private half-',1,inplace=True)
    data['bathrooms_text'].replace('1.5',2,inplace=True)
    x=data['neighbourhood_cleansed'].value_counts()
    propertyTypeList=[]
    for propertType in x.keys():
        propertyTypeList.append(propertType)
    index=1
    
    for semt in propertyTypeList:
        data['neighbourhood_cleansed'].replace(semt,index,inplace=True)
        index+=1
    
    x=data['room_type'].value_counts()
    propertyTypeList=[]
    for propertType in x.keys():
        propertyTypeList.append(propertType)
    index=1
    print( x.keys())
    
    for semt in propertyTypeList:
        data['room_type'].replace(semt,index,inplace=True)
        index+=1
    saveDataChange(data)

    
    x=data['host_response_time'].value_counts()
    propertyTypeList=[]
    for propertType in x.keys():
        propertyTypeList.append(propertType)
    index=1
    
    for semt in propertyTypeList:
        data['host_response_time'].replace(semt,index,inplace=True)
        index+=1
    return data

def saveDataChange(data):
    data.to_csv("CleanedData.csv",index=False)

