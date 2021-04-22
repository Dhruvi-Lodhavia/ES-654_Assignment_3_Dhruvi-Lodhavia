from FC_NN import FC_layer
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
from autograd import grad
from autograd import elementwise_grad as egrad
import autograd.numpy as jnp
import numpy as np
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
from sklearn.model_selection import KFold
def load_dataset1():
    
    scaler = MinMaxScaler()
    #10 classes
    
    #loading X and y values in dataframes
    digits = load_digits(n_class=10)  # consider binary case
    X = digits.data
    X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(digits.target)
    n_classes = np.unique(y)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X,y,n_classes

#calculates forward propogation and then call loss(used to calculate derivative of loss wrt bias and weights)
def forwardprop_loss(weights, bias, X_train, y_train,network):
    input = X_train
    for i in range(len(network)):
        #forward propogation
        Z = jnp.dot(input, jnp.array(weights[i])) + jnp.array(jnp.tile(bias[i],(input.shape[0],1)))
        A = network[i].get_activation_function(network[i].activation_fn)(Z)
        input = A
    #calling cross entropy loss 
    loss = network[-1].cross_entropy_loss(A,y_train,len(n_classes))
    return loss

#predicting by using forward propogation and calculated finalized weights
def predict(network, input_layer):
    for layer in network:
        input_layer = layer.forward(input_layer)
    index = np.argmax(input_layer,axis=1)
    return index
#getting test and train dataset
X,y,n_classes = load_dataset1()
accuracy_list = []
k_fold = KFold(n_splits=3,shuffle=False)
k_fold.split(X)    

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test,y_train, y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
    #defining two layers, 1 sigmoid and another softmax

    network = [
    FC_layer((X_train.shape[0],X_train.shape[1]), (X_train.shape[0],20),'sigmoid'),
    FC_layer((X_train.shape[0],20),(X_train.shape[0],len(n_classes)),'softmax')
    ]
    #running for 400 epoch
    epochs =400
    learning_rate = 1

    for i in trange(epochs):
        j= 0
        weights = []
        bias=[]
        #parsing each layer of network fro forward 
        #saving weights and biases
        for layer in network:
            if j==0:
                A_value = layer.forward(X_train)
                weights.append(layer.weights)
                bias.append(layer.bias)
            else:
                A_value = layer.forward(A_value)
                weights.append(layer.weights)
                bias.append(layer.bias)
            j+=1

        #calculating derivative of loss wrt weight
        dL_dw = egrad(forwardprop_loss,0)(weights,bias,X_train,y_train,network)
        #calculating derivative of loss wrt bias
        dL_db = egrad(forwardprop_loss,1)(weights,bias,X_train,y_train,network)
        #updating weights and biases
        for k in range(len(network)):
            network[k].weights -= learning_rate * dL_dw[k]/len(X_train)
            network[k].bias -= learning_rate * dL_db[k]/len(X_train)


    y_hat = predict(network,X_test)

    acc = np.mean(y_hat == y_test)
    accuracy_list.append(acc)
  
for i in range(3):
    print("accuracy of fold "+str(i)+" is "+str(accuracy_list[i]))
