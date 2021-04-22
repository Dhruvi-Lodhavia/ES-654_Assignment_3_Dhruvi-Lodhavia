import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from tqdm import trange
np.random.seed(42)


# dataset functiom to create a random dataset with N rows and P columns
def dataset(N,P):
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(2, size=N))
    return X,y
# ************************************************************************
# varying N
# empty list to store values of N in LR and Normal equation to store values
varN_LR = []
varN_N = []
P = 50
for n in trange(1,500): # varying number of samples
    X,y = dataset(n,P)
    begin_LR = 0
    end_LR = 0
    begin_N = 0
    end_N = 0
    # using LR and finding start and end time in fitting
    for i in range(5):
        LR = LinearRegression()
        begin_LR += time.time() #start and end for linear regression fitting
        LR.logistic_non_regularized(X, y,len(y),20) 
        end_LR += time.time()

    for i in range(5):
        begin_N = time.time() #start and end for predict
        LR.predict(X) 
        end_N = time.time()

    # appeding the values
    varN_LR.append((end_LR-begin_LR)/5)
    varN_N.append((end_N-begin_N)/5)
N = range(1,500)
fig1= plt.figure(1)

plt.plot(N,varN_LR)
plt.plot(N,varN_N)
plt.xlabel("N")
plt.legend(["Linear regression fitting time", "Predicting"])
fig1.savefig("./timevsN.png")


# *************************************************************************
# varP_LR =[]
# varP_N =[]
# N = 300
# for p in trange(1,300): # varying number of features 
#     X,y = dataset(N,p)
#     begin_LR = 0
#     end_LR = 0
#     begin_N = 0
#     end_N = 0
#     for i in range(5):
#         LR = LinearRegression(fit_intercept=False)
#         begin_LR += time.time() #start and end for linear regression fitting
       
#         LR.logistic_non_regularized(X, y,len(y),20) 
#         end_LR += time.time()

    
#         begin_N += time.time() #start and end for normal equation
#         LR.predict(X) 
#         end_N += time.time()

#     varP_LR.append((end_LR-begin_LR)/5)
#     varP_N.append((end_N-begin_N)/5)
# P = range(1,300)
# fig1= plt.figure(1)

# plt.plot(P,varP_LR)
# plt.plot(P,varP_N)
# plt.legend(["Logistic regression fitting time", "predict"])
# plt.xlabel("P")
# fig1.savefig("./timevsP.png")
# # # ***********************************************************
# # varying iterations
# variter_LR =[]
# variter_N =[]
# N = 300
# P = 30
# X,y = dataset(N,P)
# for iter in trange(1,100): # varying maximum iterations
#     begin_LR = 0
#     end_LR = 0
#     begin_N = 0
#     end_N = 0
#     X,y = dataset(N,P)
#     for i in range(5):
#         LR = LinearRegression()
#         begin_LR += time.time() #start and end for linear regression fitting
#         LR.logistic_non_regularized(X, y,len(y),iter)
#         end_LR += time.time()

#     for i in range(5):
#         begin_N += time.time() #start and end for normal method
#         LR.predict(X) 
#         end_N += time.time()


#     variter_N.append((end_N-begin_N)/5)
#     variter_LR.append((end_LR-begin_LR)/5)


# i = range(1,100)


# fig1= plt.figure(1)

# plt.plot(i,variter_LR)
# plt.plot(i,variter_N)
# plt.legend(["Logistic regression fitting time", "predict time"])
# plt.xlabel("iterations")



# fig1.savefig("./timevsiterations.png")

