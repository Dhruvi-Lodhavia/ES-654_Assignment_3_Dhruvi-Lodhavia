
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

print("\n*************************Q2_A-using L1***************************")
LR = LinearRegression(fit_intercept=True)
LR.fit_L1_norm(X, y,len(X),lambda_val=0.01) 
y_hat = LR.predict(X)
print('accuracy: ', accuracy(y_hat, y))

#Q1_b
print("\n*************************Q2_A-using L2***************************")
LR = LinearRegression(fit_intercept=True)
LR.fit_L2_norm(X, y,len(X),lambda_val=0.01) 
y_hat = LR.predict(X)
print('accuracy: ', accuracy(y_hat, y))