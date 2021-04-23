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
def load_dataset2():
    
    scaler = MinMaxScaler()
    #loading X and y values in dataframes
    boston = load_boston()  
    X = boston.data
    X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(boston.target)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X,y

#boston dataset
X,y = load_dataset2()

#calculates forward propogation and then call loss(used to calculate derivative of loss wrt bias and weights)
def forwardprop_loss2(weights, bias, X_train, y_train,network):
    input = X_train
    for i in range(len(network)):
        #forward propogation
        Z = jnp.dot(input, jnp.array(weights[i])) + jnp.array(jnp.tile(bias[i],(input.shape[0],1)))
        A = network[i].get_activation_function(network[i].activation_fn)(Z)
        input = A
    #calling cross entropy loss 
    loss = network[-1].rmse(A,y_train)
    return loss

#predicting by using forward propogation and calculated finalized weights
def predict2(network, input_layer):
    for layer in network:
        input_layer = layer.forward(input_layer)
    A = input_layer
    error = (np.square(np.subtract(A,y_test.reshape(-1,1))).mean())**0.5
    return error



mse = []
k_fold = KFold(n_splits=3,shuffle=False)
k_fold.split(X)    

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test,y_train, y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
    #defining two layers
    network = [
        FC_layer((X_train.shape[0],X_train.shape[1]), (X_train.shape[0],20),'relu'),
        FC_layer((X_train.shape[0],20),(X_train.shape[0],1),'relu'),
    ]
    #running for 300 epoch
    epochs = 300
    learning_rate = 3
    for i in trange(epochs):
        j= 0
        weights = []
        bias=[]
        #parsing each layer of network for forward 
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
        dL_dw = egrad(forwardprop_loss2,0)(weights,bias,X_train,y_train,network)
        #calculating derivative of loss wrt bias
        dL_db = egrad(forwardprop_loss2,1)(weights,bias,X_train,y_train,network)
        for i in range(len(network)):
            network[i].weights -= learning_rate * dL_dw[i]/len(X_train)
            network[i].bias -= learning_rate * dL_db[i]/len(X_train)
        

    error = predict2(network,X_test)
    
    mse.append(error)
    print('rmse: %.4f' % error)

for i in range(3):
    print("mse of fold "+str(i)+" is "+str(mse[i]))