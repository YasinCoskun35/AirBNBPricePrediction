import pandas as pd
from Functions import DataCleaning as dc
from Functions import Utilities as util
data=pd.read_csv('listings.csv',header=0)
testData=dc.checkColumnUsability(data)
trainData=util.prepareTestData(testData)
print(util.TestAnova(trainData))