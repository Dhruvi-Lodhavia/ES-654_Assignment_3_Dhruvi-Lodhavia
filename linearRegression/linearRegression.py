# import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
import autograd.numpy as np
import math
# from tqdm import trange


class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.y_hat = None
        self.X_1 = None
        self.Y_1 =None
        self.theta0 = []
        self.theta1 = []
        self.iter = 0
        self.rss_error =[]
        self.grad1_arr =[]
        self.grad2_arr =[]
        self.lambda_val = None
        self.labels = None
        pass

    def logistic_non_regularized_non_vectorized(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
       
       # if intercept is true, add a coloumn of ones at the start of X 
        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
        # xmini and y mini will contain mini batches
        X_mini =[] 
        y_mini =[]
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        theta = np.random.rand(m, 1)
        theta = np.zeros(len(X.columns)).flatten()
        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            theta_copy = theta.copy()
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            # looping over the columns
            for i in range(m):
                # J_der is the gradient
                J_der=0
                # looping over the batch
                for j in range(batch_size):
                    # hypo is the hypothesis ie theta1*X[0] + theta2*X[1] ....
                    hypo = 0
                    for k in range(m):
                        hypo += theta_copy[k]*X_new.iloc[j,k]
                    # calculating the gradient for the all dataset
                    #logistic regression update rules
                    J_der += (self.sigmoid(hypo)-y_new[j])*X_new.iloc[j,i]
                #updating the theta values 
                theta[i] = theta_copy[i] - lr_new*J_der/batch_size 

        self.coef_ = theta
        pass

    def logistic_non_regularized(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        # if intercept is true, add a coloumn of ones at the start of X 
        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)

        X_mini =[] 
        y_mini =[]
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        theta = np.zeros(len(X.columns)).flatten()

        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            theta_copy = theta.copy()
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            # h(theta) = X*theta
           
            
            y_pred = np.dot(X_new,theta)
            cost = (self.sigmoid(y_pred)-y_new)
            X_T = X_new.transpose()
            J_der = X_T.dot(cost)/len(y_pred)
        
            # updating theta values simulatnaeously
            theta = theta - lr_new*J_der
            # storing theta0 and theta1 values for plotting graphs
            # self.theta0.append(theta[0])
            # self.theta1.append(theta[1]) 
        self.coef_ = theta
        
    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
        # xmini and y mini will contain mini batches
        X_mini =[] 
        y_mini =[]
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        theta = np.random.rand(m, 1)
        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            theta_copy = theta.copy()
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            self.X_1 = X_new
            self.Y_1 = y_new
            # h(theta) = X*theta
            # h_theta = np.dot(X_new,theta)
            theat = np.array(theta)
            X_T = X_new.transpose()
            #calcualating gradient of cost function using autograd
            gradient_calc = grad(self.new_cost_Function)
            #passing theta to cost function
            gradient_final = gradient_calc(theat)
            gradient_final =np.array([0 if math.isnan(i) else i for i in gradient_final])
            # updating theta values simulatnaeously
            # print(gradient_final)
            theta = theta - lr_new*gradient_final/batch_size
            # storing theta0 and theta1 values for plotting graphs
            self.theta0.append(theta[0])
            self.theta1.append(theta[1]) 
        self.coef_ = theta
        pass
    
    

    def fit_k_class_autograd(self, X, y, batch_size, n_iter=100, lr=0.0001, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        if(self.fit_intercept == True):
        #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)

        X_mini =[] 
        y_mini =[]
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        unique_classes = len(np.unique(y))
        self.labels = np.array(list(set(y)))
        theta = np.zeros((m,unique_classes))

        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            theta_copy = theta.copy()
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            self.X_1 = X_new
            self.Y_1 = y_new
            # h(theta) = X*theta
            h_theta = np.dot(X_new,theta)
            X_T = X_new.transpose()
            #calcualating gradient of cost function using autograd
            gradient_calc = grad(self.cost_function_cross_Entropy)
            #passing theta to cost function
            gradient_final = gradient_calc(theta)
            # gradient_final =np.array([0 if math.isnan(i) else i for i in gradient_final ])
            # updating theta values simulatnaeously
            theta = theta - lr_new*gradient_final/batch_size
            # storing theta0 and theta1 values for plotting graphs
            self.theta0.append(theta[0])
            self.theta1.append(theta[1]) 
        self.coef_ = theta
        pass

    def fit_L1_norm(self, X, y, batch_size, n_iter=100, lr=0.01,lambda_val=1, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.lambda_val = lambda_val
        if(self.fit_intercept == True):
        #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)

        X_mini =[] 
        y_mini =[]
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        # theta = np.zeros(len(X.columns)).flatten()
        theta = np.random.rand(m, 1)
        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            theta_copy = theta.copy()
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            self.X_1 = X_new
            self.Y_1 = y_new
            # h(theta) = X*theta
            h_theta = np.dot(X_new,theta)
            X_T = X_new.transpose()
            #calcualating gradient of cost function using autograd
            gradient_calc = grad(self.new_cost_Function_L1_norm)
            #passing theta to cost function
            gradient_final = gradient_calc(theta)
            gradient_final =np.array([0 if math.isnan(i) else i for i in gradient_final ])
            # updating theta values simulatnaeously
            theta = theta - lr_new*gradient_final/batch_size
            # storing theta0 and theta1 values for plotting graphs
            self.theta0.append(theta[0])
            self.theta1.append(theta[1]) 
        self.coef_ = theta
        pass

    def fit_L2_norm(self, X, y, batch_size, n_iter=100, lr=0.01,lambda_val=1, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.lambda_val = lambda_val
        if(self.fit_intercept == True):
        #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)

        X_mini =[] 
        y_mini =[]
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        theta = np.zeros(len(X.columns)).flatten()

        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            theta_copy = theta.copy()
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            self.X_1 = X_new
            self.Y_1 = y_new
            # h(theta) = X*theta
            h_theta = np.dot(X_new,theta)
            X_T = X_new.transpose()
            #calcualating gradient of cost function using autograd
            gradient_calc = grad(self.new_cost_Function_L2_norm)
            #passing theta to cost function
            gradient_final = gradient_calc(theta)
            gradient_final =np.array([0 if math.isnan(i) else i for i in gradient_final ])
            # updating theta values simulatnaeously
            theta = theta - lr_new*gradient_final
            # storing theta0 and theta1 values for plotting graphs
            self.theta0.append(theta[0])
            self.theta1.append(theta[1]) 
        self.coef_ = theta
        pass

    def fit_k_class(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
       
       # if intercept is true, add a coloumn of ones at the start of X 
        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
        # xmini and y mini will contain mini batches
        X_mini =[] 
        y_mini =[]
        unique_classes = len(np.unique(y))
        self.labels = np.array(list(set(y)))
        # print(self.labels)
        m = len(X.columns) #number of coloumns
        n = len(X) # number of rows
        batches = n//batch_size # number of mini batches to be created
        b = int(batches) 
        batch_size = int(batch_size) 
        for i in range(0,b): # creating such mini batches
            X_mini.append(X.iloc[i*batch_size:(i+1)*batch_size])
            y_mini.append(y.iloc[i*batch_size:(i+1)*batch_size])
        # intializing random thetas
        theta = np.zeros((m,unique_classes))
        iter =1
        lr_new = lr
        # iterations
        for i in range(n_iter):
            #if lr type is inverse change lr after every iteration
            # theta copy is created for every iteration
            if(lr_type !='constant'):
                lr_new = lr/iter
                iter+=1
            # choosing the batches according to iteration
            X_new = X_mini[int(i%batches)].reset_index(drop =True)
            y_new = y_mini[int(i%batches)].reset_index(drop =True)
            # looping over the columns

            for cls_label in range(unique_classes):
                cost=0  
                theta_copy = theta.copy()

                I = (y_new == self.labels[cls_label]).astype(float)
               
                val = self.softmax(X_new,cls_label,theta)
                cost = cost - X_new.T.dot(I-val)
                theta[:,cls_label] = theta[:,cls_label]-lr*cost/batch_size # updating theta  
        
        self.coef_= theta
      
    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        # if intercept is true, add a coloumn of ones at the start of X 
        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
         
        
        X = np.array(X)
        y = np.array(y)
        # check if matrix is invertible or give and error
        a = np.linalg.matrix_rank(X.T.dot(X))-X.shape[1]
        assert (a==0), "Matrix is not invertible"
        # calculate theta using normal equation
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.coef_=theta
        return None
      
    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        
        # if intercept is true, add a coloumn of ones at the start of X 
        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
        # y_hat is X*theta
        # print(X.shape)
        # print(self.coef_.shape)
        y_hat = pd.Series(np.dot(X,self.coef_).flatten())
        self.y_hat = pd.Series([1 if i>0  else 0 for i in y_hat])
        return self.y_hat
        pass

    def predict_k_class(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        
        # if intercept is true, add a coloumn of ones at the start of X 
        if(self.fit_intercept == True):
            #initializing coloumns of 1
            new_X_ones = np.ones(len(X))
            #converting it into pd.Series
            new_X_ones = pd.Series(new_X_ones)
            #addding it to X
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
        # y_hat is X*theta
        n= len(X)
        y_hat = np.zeros(n)
        unqiue_classes=len(self.labels)

        Prob = np.zeros((n,unqiue_classes))
        for i in range(self.coef_.shape[1]):
            Prob[:,i]= self.softmax(X, i,self.coef_)
    
        y_hat=self.labels[np.argmax(Prob,axis=1)]
        y_hat=pd.Series(y_hat)
        return y_hat
       
    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        
        lw = 0
        ul = 25
        tot = abs(lw)+abs(ul) # total points
        surface = np.zeros((2*tot,2*tot)) # surface numpy array filled with zeros 
        ones = pd.Series(np.ones(len(X)))  # pd series of ones 
        error = t_0*ones + t_1*X.iloc[:,0] # calculating loss
        rss_errors = np.mean(np.square(error-y)) # calculating the error by finding the square and finding mean
        self.rss_error.append(rss_errors) # rss_errors are saved  

        # plotting a 3d graph
        ax = plt.axes(projection='3d')
        # setting x and y limits
        ax.set_xlim(lw,ul)
        ax.set_ylim(lw,ul) 

        # to save points of surface wrt t0 and t1
        for t0 in range(0,2*tot):
            for t1 in range(0,2*tot):
                error_val = (t0/2+lw)*ones+(t1/2+lw)*X.iloc[:,0]
                rss = np.sum(np.square(error_val-y))
                surface[t0][t1]= rss
        t0 = np.linspace(lw,ul,2*tot)
        t1 = np.linspace(lw,ul,2*tot)
        
        # plotting all previous points
        ax.scatter(self.theta1[:self.iter],self.theta0[:self.iter],self.rss_error[:self.iter],color="r")  
        self.iter+=1
        # plotting the surface
        ax.plot_surface(t1, t0, surface,cmap="viridis")
        
        ax.view_init(50,80)
        plt.xlabel("theta1")
        plt.ylabel("theta0")
        ax.set_title("rss_error ="+str(rss_errors))
        return ax


        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
       
        # ti =[t_0,t_1]
        y_pred=[t_1*X.iloc[j,0] + t_0 for j in range(len(X))] 
        plt.figure()
        plt.plot(X.iloc[:,0],y_pred)
        plt.scatter(X.iloc[:,0],y)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("t_1: "+str(t_1)+", t_0: "+str(t_0))
        return plt
        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        
       
        # upper and lower x and y limits for plotting
        lw = -25
        ul = 25
        tot = abs(lw)+abs(ul) # total points
       

        
        ones = pd.Series(np.ones(len(X))) 
        X = pd.concat([ones,X],axis=1, ignore_index=True)
        # contour surface is filled with 0 initially
        contour_surface = np.zeros((tot,tot))
        
        # finding y_pred
        points = t_0*ones + t_1*X.iloc[:,1]
        loss = points-y
        # calculating gardient wrt theta1 and theta 0
        xdash = X.transpose()
        grad1 = (xdash.dot(loss)/len(points))[0]
        grad2 = (xdash.dot(loss)/len(points))[1]

        

        for t0 in range(0,tot):
            for t1 in range(0,tot):
                point = (t0+lw)*ones+(t1+lw)*X.iloc[:,1]
                cont = np.sum(np.square(point-y))
                contour_surface[t0][t1]= cont
        t0 = np.linspace(lw,ul,tot)
        t1 = np.linspace(lw,ul,tot)
        plt.figure()
        # setting x and y limits of axis
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.xlim(lw,ul)
        plt.ylim(lw,ul)
        plt.contour(t1, t0, contour_surface,cmap='viridis')
        plt.annotate('', xy=(t_1,t_0), xytext=(t_1+grad2,t_0+grad1),annotation_clip=True,
                      arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                      va='center', ha='center')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("t_1: "+str(t_1)+", t_0: "+str(t_0))
        return plt        
        pass

    def max_theta(self):
        theta = self.coef_
        # max_thet = abs(np.amax(theta))
        return theta

    def cost_Function(self,theta):
        X = np.array(self.X_1)
        y = np.array(self.Y_1)
        hypothesis = np.dot(X,theta)
        cost = np.sum(np.square(y-hypothesis))/len(y)
        return cost

    def create_mini_batches(self,batch_size):
        X = self.X_1
        y = self.Y_1
        pass

    def theta_vals(self):
        return(self.theta0,self.theta1)

    def sigmoid(self,z):
        return 1/(1+np.exp(-np.array(z,dtype=float)))

    def new_cost_Function(self,theta):
        X = np.array(self.X_1)
        y = np.array(self.Y_1)
        X_theta = np.dot(X,theta)
        hypothesis = self.sigmoid(X_theta)
        cost = -np.sum((np.dot(y.T,np.log(hypothesis)))+(np.dot((np.ones(y.shape)-y).T,np.log(np.ones(y.shape)-hypothesis))))
        return cost

    def new_cost_Function_L1_norm(self,theta):
        X = np.array(self.X_1)
        y = np.array(self.Y_1)
        X_theta = np.dot(X,theta)
        hypothesis = self.sigmoid(X_theta)
        cost = -np.sum((np.dot(y.T,np.log(hypothesis)))+(np.dot((np.ones(y.shape)-y).T,np.log(np.ones(y.shape)-hypothesis))))
        cost_ans = cost + self.lambda_val*np.sum(np.abs(theta))
        return cost_ans

    def new_cost_Function_L2_norm(self,theta):
        X = np.array(self.X_1)
        y = np.array(self.Y_1)
        X_theta = np.dot(X,theta)
        hypothesis = self.sigmoid(X_theta)
        cost = -np.sum((np.dot(y.T,np.log(hypothesis)))+(np.dot((np.ones(y.shape)-y).T,np.log(np.ones(hypothesis.shape)-hypothesis))))
        sq_rt = np.array(theta*theta.transpose())[0]
        
        cost_ans = cost + (self.lambda_val)*np.array(theta*(theta.transpose()))[0]
        return cost_ans

    def softmax(self,X,cls_label,theta):
        den = np.exp(np.dot(X,theta))
        return den[:,cls_label]/np.sum(den,axis=1)
        
    def cost_function_cross_Entropy(self,theta):

        X = np.array(self.X_1)
        y = np.array(self.Y_1)
        den = np.exp(np.dot(X,theta)) 
        prob = den/np.sum(den,axis=1).reshape(-1,1)
        cost = 0
        for i in range(len(self.labels)):
            cost = cost - np.dot(np.array(y == i,dtype =float),np.log(prob[:,i]))     
        return np.array(cost)

    def plot_decision_boundary(self, X, y,i,j,name1,name2):
        # print(self.coef_.shape)
        # print(X.shape)
        fig = plt.figure()
        intercept,feat1,feat2 = self.coef_[0],self.coef_[i],self.coef_[j]

        m = -feat1/feat2
        intercept /= -feat2
        xmin, xmax, ymin, ymax = -2,2,-1,1

        Xs = np.array([xmin, xmax])
        ys = m*Xs + intercept
        
        plt.plot(Xs, ys, 'k', lw=1, ls='--')
        plt.fill_between(Xs, ys, ymin, color='tab:red', alpha=0.2)
        plt.fill_between(Xs, ys, ymax, color='tab:blue', alpha=0.2)
        plt.scatter(X[y==0][0],X[y==0][1],s=8,alpha=0.5,cmap='Paired',label = "0")
        plt.scatter(X[y==1][0],X[y==1][1],s=8,alpha=0.5,cmap='Paired',label = "1")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(name2)
        plt.xlabel(name1)
        plt.savefig("Decision_boundary.png")