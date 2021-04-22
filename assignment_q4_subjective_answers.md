# ES654-2021 Assignment 2

Dhruvi Lodhavia* - *18110050*

-------
 N = number of samples,
 P = number of attributes,
 t = number of iterations,
 k = number of classes

## Time complexity

- Time complexity of gradient descent is 
O(NPt).
Therefore, for k classes
- The time complexity of fitting logistic regression can be written as 
O(P*N*t*k)

- The time complexity for this is O(N * P * k) since we are essentially multiplying our X i.e NP with thetas which is P*k
- Time complexity of prediction for logistic regression is O(P*N*k) as we are performing the operation of X*theta which is (N,P)x(P,1) for k classes


## Space complexity
- For fitting, space complexity is O(N*P) + O(P*k) + O(N*k)
for storing X,theta and X*theta

- For predicting, space complexity is O(N*P)+(P*k)+ for storing X and theta


>Varying N and fitting time for P = 50

- With varying N, both should vary linearly(increases slowly)
![varying N](\timevsN.png)

>Varying P and fitting time for N = 300

- as P increases the time complexity increases linearly as expected
![varying P](\timevsP.png)

>Varying iterations and fitting time for N=300  ,P= 30

- in gradient descent, the graph increases linearly with time as expected

![varying iterations](\timevsiterations.png) 
