import pandas as pd
import numpy as np

def fill_data():
    data=pd.read_csv("listings_cleansed.csv")
    data['minimum_nights'].fillna(data['minimum_nights'].mean())
    data['maximum_nights'].fillna(data['maximum_nights'].mean())
    data['host_is_superhost'].fillna('f',inplace=True)
    data.dropna(subset=['neighbourhood_cleansed','room_type'],inplace=True)
    data['bathrooms_text'].fillna('1',inplace=True)
    data['review_scores_accuracy'].fillna('5',inplace=True)
    data['review_scores_cleanliness'].fillna('5',inplace=True)
    data['review_scores_communication'].fillna('5',inplace=True)
    data['review_scores_checkin'].fillna('5',inplace=True)
    data['review_scores_location'].fillna('5',inplace=True)
    data['review_scores_value'].fillna('5',inplace=True)
    data['instant_bookable'].fillna('f',inplace=True)
    data['price']=data['price'].map(lambda x: x.lstrip('$').replace(',',''))
    data['price'] = pd.to_numeric(data['price'])
    data['price'].fillna(data['price'].mean(),inplace=True)
    
    
    data['accommodates'].fillna(data['accommodates'].mean(),inplace=True)
    numberOfNulls=data['accommodates'].isnull().sum()
    
    data.dropna(subset=['bedrooms','beds'],inplace=True)
    
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
    
    
    data['host_is_superhost'].replace('t',1,inplace=True)
    data['host_is_superhost'].replace('f',0,inplace=True)
    data['instant_bookable'].replace('t',1,inplace=True)
    data['instant_bookable'].replace('f',0,inplace=True)
    
    data['bathrooms_text']=data['bathrooms_text'].map(lambda x: str(x).rstrip('bath private shared'))
    
    
    data['bathrooms_text'].replace('Half-',1,inplace=True)
    data['bathrooms_text'].replace('Shared half-',1,inplace=True)
    data['bathrooms_text'].replace('Private half-',1,inplace=True)
    data['bathrooms_text'].replace('1.5',2,inplace=True)
    
    
    data.to_csv("listings_filled.csv")

fill_data() 