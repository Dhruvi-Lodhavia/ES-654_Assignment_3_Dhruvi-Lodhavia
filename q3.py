import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from tqdm import trange
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import itertools

warnings.filterwarnings("ignore",category =RuntimeWarning)
np.random.seed(42)

N = 10
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series([1,0,0,1,0,0,1,0,0,1])

print("\n*************************Q3_A-update_rules***************************")
LR = LinearRegression(fit_intercept=True)
LR.fit_k_class(X, y,len(X)) 
y_hat = LR.predict_k_class(X)
print('accuracy: ', accuracy(y_hat, y))

#Q1_b
print("\n*************************Q3_B-using autograd***************************")
LR = LinearRegression(fit_intercept=True)
LR.fit_k_class_autograd(X, y,len(X)) 
y_hat = LR.predict_k_class(X)
print('accuracy: ', accuracy(y_hat, y))








from sklearn.datasets import load_digits
data,target = load_digits(return_X_y=True,as_frame=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


Xy = pd.concat([data, target.rename("y")],axis=1, ignore_index=True)
Xy_shuffled = np.arange(len(Xy)) #shuffling the dataset
np.random.shuffle(Xy_shuffled)
Xy_new = Xy.iloc[Xy_shuffled].reset_index(drop=True)

accuracy_cv = []  
# function defined that returns data in parts(4 and 1) depending on k value given
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
    # dividing test and train into 5 different datasets
    train,test = threefoldcvdataset(Xy_new,k)
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]].astype('category')
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]].astype('category')
    # calling decision tree
    LR = LinearRegression(fit_intercept=False)
    new = LR.fit_k_class_autograd(X_train,y_train,len(X_train)/5)
    y_hat = LR.predict_k_class(X_test)
    # returning accuracy found and appending in
    accuracy_cv.append(accuracy(y_hat, y_test))
    # print(accuracy(y_hat,y_test))
    
# returing max accuracy
m_acc = sum(accuracy_cv)/3
print("accuracy is ",m_acc)
    
#printing the confusion matrix
print(confusion_matrix(y_test, y_hat))

#plotting the confusion matrix
#source - stackoverflow
def plot_confusion_matrix(conf,target_names,title='Confusion matrix',cmap=None,normalize=True): 

    accuracy = np.trace(conf) / float(np.sum(conf))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(conf, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    if normalize:
        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]


    thresh = conf.max() / 1.5 if normalize else conf.max() / 2
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(conf[i, j]),
                     horizontalalignment="center",
                     color="white" if conf[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(conf[i, j]),
                     horizontalalignment="center",
                     color="white" if conf[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
#plotting the matrix
plot_confusion_matrix ( confusion_matrix(y_test, y_hat), 
                      normalize    = False,
                      target_names = ['0','1', '2', '3','4', '5', '6','7', '8', '9'],
                      title        = "Confusion Matrix")  
#principal componenet analysis
pca = PCA(n_components=2)
pca_trans = pca.fit_transform(data)
#plotting the numbers
plt.scatter(pca_trans[:, 0], pca_trans[:, 1], c=target, cmap="Dark2")
plt.colorbar()
plt.show()

###