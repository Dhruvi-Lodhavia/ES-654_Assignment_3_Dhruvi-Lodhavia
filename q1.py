
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from tqdm import trange
np.random.seed(42)

N = 10
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series([1,0,0,1,0,0,1,0,0,1])

#Q1_a

print("\n*************************Q1_A-update_rules***************************")
LR = LinearRegression(fit_intercept=True)
LR.logistic_non_regularized(X, y,len(X)) 
y_hat = LR.predict(X)
print('accuracy: ', accuracy(y_hat, y))

#Q1_b
print("\n*************************Q1_B-using autograd***************************")
LR = LinearRegression(fit_intercept=True)
LR.fit_autograd(X, y,len(X)) 
y_hat = LR.predict(X)
print('accuracy: ', accuracy(y_hat, y))

#Q1_c
print("\n*************************Q1_C-breast cancer dataset***************************")


data,target = load_breast_cancer(return_X_y=True,as_frame=True)
# print(data.feature_names)
#normalizing the data between 0-1
data = (data-data.min())/(data.max()-data.min())
# print(data.shape)
Xy = pd.concat([data, target.rename("y")],axis=1, ignore_index=True)
Xy_shuffled = np.arange(len(Xy)) #shuffling 
np.random.shuffle(Xy_shuffled)
Xy_new = Xy.iloc[Xy_shuffled].reset_index(drop=True)
# print(Xy_new.shape)

accuracy_cv = []  
# function defined that returns data in parts(2 and 1) depending on k value given
def threefoldcvdataset(XY,k):
    l = int(len(XY)/3)
    
    test = XY[l*k:l*(k+1)]
    test = test.reset_index(drop=True)
    # if test part is first
    if(k==0):
        train = XY[l:]
    else:
        train_p1 = XY[0:l*k]
        train_p2 = XY[l*(k+1):]
        train = pd.concat([train_p1,train_p2],axis=0)
    # returning teh two parts
    train = train.reset_index(drop=True)
    return train,test

for k in trange(3):
    # dividing test and train into 3 different datasets
    train,test = threefoldcvdataset(Xy_new,k)
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]].astype('category')
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]].astype('category')
    # calling decision tree
    LR = LinearRegression(fit_intercept=False)
    LR.fit_autograd(X_train,y_train,len(X_train))
    y_hat = LR.predict(X_test)
    if(k==1):
        # print(X.shape)
        data1 = load_breast_cancer()
        name1= data1.feature_names[5]
        name2 = data1.feature_names[10]
     
        LR.plot_decision_boundary(X_train,y_train,5,10,name1,name2)
    # returning accuracy found and appending in 
    accuracy_cv.append(accuracy(y_hat, y_test))
    # print(accuracy(y_hat,y_test))
    
# returing max accuracy
m_acc = sum(accuracy_cv)/3
print("accuracy is ",m_acc)
    


    
