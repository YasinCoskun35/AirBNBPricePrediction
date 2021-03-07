import pandas as pd

data=pd.read_csv('normalizedData.csv')
max=data.price.max()
categories=[]
i=0
categoryNumber=1
while i<=max:
    categories.append({'CategoryNumber':categoryNumber,'MinPrice':i+1,'MaxPrice':i+100})
    i+=100
    categoryNumber+=1

def setPriceCategory(price):
    for category in categories:
        if (price>=category['MinPrice'] and price<=category['MaxPrice'] ):
            print('Category Number : '+str(category['CategoryNumber'])+'  Price: '+str(category['MaxPrice']))
            return category['CategoryNumber']
data=data.drop(columns="beds")

data['price']=data['price'].map(lambda x: setPriceCategory(x))
data.to_csv('enSonHali.csv')
