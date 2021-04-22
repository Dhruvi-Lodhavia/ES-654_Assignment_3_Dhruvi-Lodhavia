import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from tqdm import trange
import warnings
np.random.seed(42)

warnings.filterwarnings("ignore",category =RuntimeWarning)

data,target = load_breast_cancer(return_X_y=True,as_frame=True)
#normalizing the data between 0-1
data = (data-data.min())/(data.max()-data.min())

Xy = pd.concat([data, target.rename("y")],axis=1, ignore_index=True)
Xy_shuffled = np.arange(len(Xy)) #shuffling the dataset
np.random.shuffle(Xy_shuffled)
Xy_new = Xy.iloc[Xy_shuffled].reset_index(drop=True)


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


# # ris = pd.read_csv("Iris.csv", header = None, names = ["sepal length", "sepal width", "petal length", "petal width", "label"])
# best_lamda = 0
# lam = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
# accuracy_list = []
# max_acc = 0
# # running nested cv for 5 loops
# for fold in trange(3):
#     # dividing dataset into test and trai n
#     train,test = threefoldcvdataset(Xy_new,fold)
#     X_train = train[train.columns[:-1]]
#     y_train = train[train.columns[-1]].astype('category')
#     X_test = test[test.columns[:-1]]
#     y_test = test[test.columns[-1]].astype('category')
#     XY = pd.concat([X_train,y_train],axis = 1)   
#     for lambda_val in lam:
#         #varying lambda_val
#         acc = []
#         for k in range(3):
#             # varying folds for validation set and dividing train set in test and validate and fitting it in the tree
#             train1,valid = threefoldcvdataset(XY,k)
#             X_train1 = train1[train1.columns[:-1]]
#             y_train1 = train1[train1.columns[-1]].astype('category')
#             X_valid = valid[valid.columns[:-1]]
#             y_valid = valid[valid.columns[-1]].astype('category')

#             LR = LinearRegression(fit_intercept=True)

#             LR.fit_L1_norm(X_train1, y_train1,len(X_train1),lambda_val=lambda_val) # here you can use fit_non_vectorised / fit_autograd 
#             # y_hat = LR.predict(X)
#             # tree = DecisionTree(criterion='information_gain',max_lambda_val = lambda_val+1)
#             # tree.fit(X_train1, y_train1)
#             y_hat = LR.predict(X_valid)
#             # appending accuracy for each iteration
#             acc.append(accuracy(y_hat, y_valid))
#         # calc avg for each k
#         avg_acc = sum(acc)/len(acc)
#         # finding max acc for each lambda_val and its values
#         if(max_acc < avg_acc):
#             max_acc = avg_acc
#             best_lamda = lambda_val
#         # lam.append(lambda_val)
#         accuracy_list.append(avg_acc)
# for i in range(5):
#     print("accuracy is",accuracy_list[i],"for lambda_val",lam[i])



thetas=[]
m = 0.1
ld=[0.001,0.005,0.01,0.05,0.1,0.5,1]
for i in ld:
    #varying ld

    LR = LinearRegression(fit_intercept=False)
    # print(data.iloc[:,:9].shape)
    # print(target.shape)
    LR.fit_L1_norm(data.iloc[:,:10],target,7,200,0.5,i)
    thetas.append(LR.max_theta())
    # ld.append(m)

count = 0
for i in np.array(thetas).transpose():
    plt.plot(ld,i.transpose(),label=str(count)) # plot for each n with legend
    count+=1


# naming the x axis 
plt.xlabel('lambda_val') 
# naming the y axis 
plt.ylabel('theta') 
# giving a title to graph 
plt.title('theta vs lambda_val') 
# function to show the plot 
plt.legend(data.iloc[:,:15])
plt.savefig("./q2_L1reg_plot.png")
plt.show()


