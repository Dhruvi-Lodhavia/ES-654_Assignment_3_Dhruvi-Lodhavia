''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=False):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree = degree
        self.include_bias = include_bias
        pass

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        # converting the numpy array to datframe
        X = pd.DataFrame(X)
        # making a copy of original X 
        X_original = X.copy()
        X_new = X.copy()
        # iterating over the coloumns and degrees until max degrees
        for col in X.columns:
            for deg in range(2,self.degree+1):
                # X_new  = X^deg
                X_new = X_original.iloc[:,col].transform(lambda x: x**deg)
                # X = X+X_new
                X = pd.concat([X,X_new],axis=1, ignore_index=True)
        
        
        #if bias is true add a column of ones at front
        if(self.include_bias==True):
            new_X_ones = np.ones(len(X))
            new_X_ones = pd.Series(new_X_ones)
            X = pd.concat([new_X_ones.rename("ones"),X],axis = 1,ignore_index=True)
        # returning numpy array as specified
        return np.array(X)
        pass
    
        
        
        
        
        
        
        
        
    
                
                
