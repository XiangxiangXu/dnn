import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

def print_for_tikz(x_list):
    x_list = x_list.reshape(-1)
    l = len(x_list)
    s = ''
    for i in range(l):
        s = s + str((i+1, x_list[i]))
    return s

def p2b(p_mat):
    p_x = np.sum(p_mat, 0)
    p_y = np.sum(p_mat, 1)
    b_mat = p_mat / np.sqrt(p_x.reshape(1, -1)) / np.sqrt(p_y.reshape(-1, 1))
    return b_mat

def get_p_hat_mat(input_data, output_data):
    """
    Estimate empirical distribution p_hat based on X, Y samples
    input_data: input, one-hot data, n x |X|
    output_data: output, one-hot data, n x |Y|
    """
    cX = input_data.shape[1]
    cY = output_data.shape[1]
    n = input_data.shape[0]
    i_val = np.dot(input_data, list(range(cX)))
    o_val = np.dot(output_data, list(range(cY)))
    p_hat_mat = np.bincount((o_val * cX + i_val).astype(int)).reshape(cY, cX) / n
    return p_hat_mat

def regulate(V, p, axis = 0, unilen = True):
    '''
    make a vector valued function s(X): Xset -> R^k
    zero mean and unit length w.r.t. p (E[s] = 0, ) 
    V is a |X| x k matrix representing s(X) = [s1(X), s2(X), ..., sk(X)]^T.
     V = [[   s1(0),    s2(0), ...,    sk(0)],
          [   s1(1),    s2(1), ...,    sk(1)],
                |         |
          [s1(cX-1), sk(cX-1), ..., sk(cX-1)]]
    p is an array, indicating the distribution of X
    axis: if axis = 1, then V is listed as V.T.
    unilen: unit length, if Ture (default), then E[s^T s] = 1.
    '''
    if len(V.shape) == 1:
        V = V - sum(V * p)
        l = np.sqrt(sum(V * V * p))
    else:
        if axis == 1:
            V = V.T
        V = V - np.matmul(p.reshape(1, -1), V)
        l = np.sqrt(np.sum(np.matmul(p.reshape(1, -1), V ** 2)))
        if axis == 1:
            V = V.T
    if unilen:
        V = V / l    
    return V, l

def info_mat(V, p, bZeromean = 1):
    """Get Infomrmation matrix from value matrix V
    V is a |X| x k matrix representing s(X) = [s1(X), s2(X), ..., sk(X)]^T.
     V = [[   s1(0),    s2(0), ...,    sk(0)],
          [   s1(1),    s2(1), ...,    sk(1)],
                |         |
          [s1(cX-1), sk(cX-1), ..., sk(cX-1)]]
    p is an array, indicating the distribution of random variable
    """
    if bZeromean:
        V = V - np.matmul(p.reshape(1, -1), V)
    V = V * np.sqrt(p).reshape(-1, 1)
    return V

def h_score(x_mat, label, b_onehot = False):
    """
    Calculate The coffiencient of Separating
    x_mat: Each row of x_mat represents an observation, and each column a single variable in these samples
    y: label( could be label or one-hot encoded label; dafault: not one-hot)
    """
    label = np.array(label)
    if b_onehot:
        label = np.array([label_c.argmax() for label_c in label])
    label_set = np.unique(label)
    x_mean = np.mean(x_mat, axis = 0) # E[X]
    cov_x = np.cov(x_mat.T)           # Cov(X)
    # pinv_cov_x = np.linalg.pinv(cov_x) # pinv(Cov(X))
    coe = 0
    """
    Here we use this formula:
    H-score = E[(E[X|Y] - E[X]) * inv(Cov(X)) * (E[X|Y] - E[X])] / 2
    """
    for label_i in label_set: # current label i
        x_i = x_mat[label == label_i, :]      # get X where corresponding Y = i
        x_i_mean = np.mean(x_i, axis = 0)     # E[X|Y = i]
        x_i_mean_c = x_i_mean - x_mean        # E[X|Y = i] - E[X]
        pr_i = float(sum(label == label_i)) / len(label)  # Pr(Y = i)
        if cov_x.size > 1:
            coe_i = np.dot(x_i_mean_c, np.linalg.solve(cov_x, x_i_mean_c))
            # np.matmul(pinv_cov_x, x_i_mean_c)
            # (E[X|Y = i] - E[X]) * inv(Cov(X)) * (E[X|Y = i] - E[X])
        else:
            coe_i = x_i_mean_c ** 2 / cov_x
        coe = coe + coe_i * pr_i
    coe = coe / 2
    return coe

def MakeLabels(X):
    '''
    return onehotencoded array
    
    Input X should be an array (nSamples, 1), taking values from alphabet 
    with size xCard
    
    X must be labels themselves, i.e. integers starting from 0, with every 
    value used. Otherwise need to use sklearn.LabelEncoder()
    
    return array of size (nSamples, xCard)
    '''    
    onehot_encoder = OneHotEncoder(sparse=False)
    temp = X.reshape(len(X), 1)
    onehots = onehot_encoder.fit_transform(temp)
    return(onehots)

def GenerateDiscreteSamples(Pxy, nSamples):
    '''    
    generate n samples of (X, Y) pairs, with distribution Pxy
    
    return a list of two np.array(dtype=int), each of size n
    values in range(cX) and range(cY)
    
    as i.i.d. sample of (X,Y) pairs with joint distribution randomly chosen
    Input:    Pxy, a cY x cX matrix

    '''        
    (yCard, xCard) = Pxy.shape
    PxyVec = Pxy.reshape(-1)  # PxyVec is the PMF of key = Y * cX + X 
    key = np.random.choice(range(xCard*yCard), nSamples, p=PxyVec)
    """
    key = Y * cX + X, Pxy[Y, X] = PxyVec[Y * cX + X], shown as follows:
    
    [[       0,            1,  ...,            cX-1], 
     [      cX,       cX + 1,  ...,       cX + cX-1],
     [     2cX,      2cX + 1,  ...,      2cX + cX-1],
            |          |       ...,          |
     [(cY-1)cX, (cY-1)cX + 1,  ..., (cY-1)cX + cX-1]]
    
    """
    Y = (key / xCard).astype(int)
    X = (key % xCard)
    
    return([X, Y])
