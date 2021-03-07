from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
def TestAnova(trainData):
    #generate dataset
    #define feature selection
    print(trainData)
    fs = SelectKBest(score_func=f_classif, k=5).fit_transform(trainData['x_train'], trainData['y_train'])
    #apply feature selection

    print(fs)
    return fs

def prepareTestData(data):
    target=data.price
    features=data.drop(columns=['price'])
    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2)
    return {'x_train':X_train,'x_test':X_test,'y_train':y_train,'y_test':y_test}