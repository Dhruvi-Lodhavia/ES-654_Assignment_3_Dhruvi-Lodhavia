import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy
from autograd import grad
from autograd import elementwise_grad as egrad
import autograd.numpy as jnp
from sklearn.metrics import accuracy_score
from tqdm import trange
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


#NN code for digits dataset

class FC_layer:
    def __init__(self, input_size, output_size,activation_fn):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size[1], output_size[1]) / np.sqrt(input_size[1] + output_size[1])
        self.bias = np.random.randn(output_size[1]) / np.sqrt(output_size[1])
        self.Z = None
        self.A = None
        self.activation_fn = activation_fn
        self.dL_dz = None
        self.dL_dw = None
        self.dL_db = None
        self.dL_dA = None
        

    def forward(self, input):
        self.input = input
        self.Z = jnp.dot(input, self.weights) + jnp.tile(self.bias,(input.shape[0],1))
        self.A = self.get_activation_function(self.activation_fn)(self.Z)
        return self.A

    def sigmoid(self,x):  
        return 1/(1+jnp.exp(-jnp.array(x,dtype=float)))

    def relu(self,x):
        return jnp.maximum(x, 0.0)
    
    def identity(self,x):
        return x
    
    def softmax(self,x):
      return jnp.exp(x)/jnp.sum(jnp.exp(x))

    def get_activation_function(self,name):

      if name=='relu':
        return self.relu
      elif name=='sigmoid':
        return self.sigmoid
      elif name=='leaky_relu':
        return self.leaky_relu
      elif name=='softmax':
        return self.softmax
      elif name=='identity':
        return self.identity
      else:
        raise ValueError('Only "relu", "leaky_relu", and "sigmoid" supported')


    def cross_entropy_loss(self, A,y,n_classes):
        loss = 0
        for i in range(n_classes):
            loss -= jnp.dot((y == i).astype(float),jnp.log(A[:,i]))   
        return loss
    
    def rmse(self,A,y):
       return jnp.sum((jnp.square(jnp.subtract(A,y.reshape(-1,1)))))/len(A)

##https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65#:~:text=FC%20layers%20are%20the%20most,connected%20to%20every%20output%20neurons









# #boston dataset
# X_train2,y_train2,X_test2,y_test2 = load_dataset2()

# def forwardprop_loss2(weights, bias, X_train, y_train,network):
#     input = X_train
#     for i in range(len(network)):
#         Z = jnp.dot(input, jnp.array(weights[i])) + jnp.array(jnp.tile(bias[i],(input.shape[0],1)))
#         A = network[i].get_activation_function(network[i].activation_fn)(Z)
#         input = A
#     loss = network[-1].rmse(A,y_train)
#     return loss


# network2 = [
#     FC_layer((X_train2.shape[0],X_train2.shape[1]), (X_train2.shape[0],20),'relu'),
#     FC_layer((X_train2.shape[0],20),(X_train2.shape[0],1),'relu'),
# ]

# epochs =300
# learning_rate = 0.1
# for i in trange(epochs):
#   j= 0
#   weights = []
#   bias=[]
#   for layer in network2:
#     if j==0:
#       A_value = layer.forward(X_train2)
#       weights.append(layer.weights)
#       bias.append(layer.bias)
#     else:
#       A_value = layer.forward(A_value)
#       weights.append(layer.weights)
#       bias.append(layer.bias)
#     j+=1


#   dL_dw = egrad(forwardprop_loss2,0)(weights,bias,X_train2,y_train2,network2)

#   dL_db = egrad(forwardprop_loss2,1)(weights,bias,X_train2,y_train2,network2)
#   for i in range(len(network2)):
#     network2[i].weights -= learning_rate * dL_dw[i]/len(X_train2)
#     network2[i].bias -= learning_rate * dL_db[i]/len(X_train2)
  

     


#   # if i%(epochs/10)==0:
#   #   print('Epoch: {}\tLoss: {:.6f}\tTrain Accuracy: {:.3f}\tTest Accuracy: {:.3f}'.format(i, loss, accuracy_score(y_train, predict(X_train, y_train, model)), accuracy_score(y_test, predict(X_test, y_test, model))))

# def predict2(network2, input_layer):
#     for layer in network2:
#         input_layer = layer.forward(input_layer)
#     return input_layer


# A = predict2(network2,X_test2)
# error = (np.square(np.subtract(A,y_test2.reshape(-1,1))).mean())**0.5
# # print('ratio: %.2f' % ratio)
# print('rmse: %.4f' % error)