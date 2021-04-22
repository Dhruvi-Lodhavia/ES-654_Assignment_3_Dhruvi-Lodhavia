def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    #
    #  assert(y_hat.size == y.size)
    # accuracy is (y_hat = y)/toal
    correct = 0.0
    
    for i in range(y_hat.size):
        if(y_hat[i] == y[i]):
            correct+=1.0
    
    total = y.size
    total = float(total)
    if(total!=0):
        return float(correct/total)
    else:
        return 0
    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    correct = 0.0
    pred = 0.0
    # precision is (y_hat == y == cls)/(y_hat == cls)
    for i in range(y.shape[0]):
        if y[i] == cls and y_hat[i] == cls:
            correct = correct + 1.0
    
    for i in range(y.shape[0]):
        if y_hat[i] == cls:
            pred = pred + 1.0
    if(pred!=0):
        return float(correct/pred)
    else:
        return 0
    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    correct = 0.0
    true = 0.0
    # precision is (y_hat == y == cls)/(y == cls)
    for i in range(y.shape[0]):
        if y[i] == cls and y_hat[i] == cls:
            correct = correct + 1.0
    
    for i in range(y.shape[0]):
        if y[i] == cls:
            true = true + 1.0
    if(true!=0):
        return float(correct/true)
    else:
        return 0
    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    err = y_hat - y
    rmse = ((err ** 2).astype('float64').mean()) ** 0.5
    
    return rmse

    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    err = y_hat - y
    mae = err.abs().astype('float64').mean()
    return mae
    pass