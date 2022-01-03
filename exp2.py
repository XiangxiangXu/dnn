"""
Feature Projection in Neural Networks with Restricted Expressive Power [Sec 3.4.2]
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras import backend as K
from func import get_p_hat_mat, regulate, p2b, info_mat, GenerateDiscreteSamples, MakeLabels, print_for_tikz
from scipy.special import logit, expit

plt.ion()

"""
Network structure:
         Layer1    Layer2
           s1  -v1-  s2  -v2- 
bias:           b1        b2
infomat:  Phi1 (W)  Phi2 Psi2

W is not an information matrix.
"""
def get_w_mat(b_mat, Phi1, Psi2, J):
    """
       Get W
    """
    inv_cov_s1 = np.linalg.pinv(np.matmul(Phi1.T, Phi1))# inv(Phi1.T * Phi1)
    inv_cov_v2 = np.linalg.pinv(np.matmul(Psi2.T, Psi2))# inv(Psi2.T * Psi2)
    Psi2_n = np.matmul(Psi2, inv_cov_v2) #Psi2 * inv(Psi2.T * Psi2)
    Phi1_n = np.matmul(Phi1, inv_cov_s1) #Phi1 * inv(Phi1.T * Phi1)
    w_mat = np.matmul(np.matmul(Psi2_n.T, b_mat), Phi1_n)
    w_mat = np.matmul(np.linalg.inv(J), w_mat)
    return w_mat

def get_s2m(b2, v2, Py):
    """
       Get theoretical value of E[s]
    """
    Psi2 = info_mat(v2, Py_hat)
    d2 = np.log(Py_hat) - b2.reshape(-1)
    xi2 = info_mat(d2.reshape(-1, 1), Py_hat)
    cov_v2 = np.matmul(Psi2.T, Psi2) # Cov(v2)
    inv_cov_v2 = np.linalg.inv(cov_v2)
    e_vd2 = np.matmul(Psi2.T, xi2) # E[v2 * d2]
    s2m = np.matmul(inv_cov_v2, e_vd2)
    return s2m

cX = 8
cY = 6
n_h = [4, 3] # nodes in hidden layers
k1 = n_h[0]
k2 = n_h[1]
"""
  [one-hot]                        [one-hot]
     cX  -  k1(tanh) - k2(sigmoid)  -  cY    
"""
nSamples = 100000  
batch_size = nSamples

np.random.seed(1)
rho = .02    # local assumption parameter, 0.02
A = np.eye(cY, cX) + 0.05 * np.random.rand(cY, cX)    
A = A * np.random.rand(cY, cX)
A = A / np.sum(A)

Prank1 = np.ones([cY, cX])/cY/cX
Prank1 = Prank1 / np.sum(Prank1)
Pxy = Prank1 * (1-rho) + rho * A
# just to get a random 'local' distribution Pxy
Px = np.sum(Pxy, axis = 0)
Py = np.sum(Pxy, axis = 1)

# generate samples (X, Y) pairs, with cardinalities cX, cY, and n Samples
[X, Y] = GenerateDiscreteSamples(Pxy, nSamples)
# convert to one-hot encoding
XLabels = MakeLabels(X)
YLabels = MakeLabels(Y)

nEpochs = 800
model = Sequential()
model.add(Dense(k1, activation='tanh', input_dim=cX))
    
model.add(Dense(k2, activation='sigmoid', input_dim=k1))     

model.add(Dense(cY, activation='softmax', input_dim=k2))
model.layers[0].trainable = False
sgd = SGD(lr=4, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=.1) 
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# with a Sequential model    
start_time = time.time()
Pxy_hat = get_p_hat_mat(XLabels, YLabels)
Bxy_hat = p2b(Pxy_hat)
Px_hat = np.sum(Pxy_hat, axis = 0)
Py_hat = np.sum(Pxy_hat, axis = 1)
U, s, V = np.linalg.svd(Bxy_hat)
s1 = K.function([model.layers[0].input], [model.layers[0].output])([np.eye(cX)])[0]
# get the value of s1.
# Return the output of the first layer given a certain input np.eye(cX)
s1m = np.matmul(Px_hat.reshape(1, -1), s1)
Phi1 = info_mat(s1, Px_hat)
plt.figure(figsize=(9, 3))

model.fit(XLabels, YLabels, verbose=0, batch_size=batch_size, epochs=nEpochs)
v2 = model.get_weights()[4].T
Psi2 = info_mat(v2, Py_hat)
b2 = model.get_weights()[5]
b20 = b2 - np.dot(b2, Py_hat)
    
s2 = K.function([model.layers[0].input], [model.layers[1].output])([np.eye(cX)])[0]
s2m = np.matmul(Px_hat.reshape(1, -1), s2)
b2t = np.log(Py_hat) - np.matmul(v2, s2m.T).reshape(-1) 
b2t = b2t - np.dot(b2t, Py_hat)
s2m_t = get_s2m(b2, v2, Py)
s2m_t = s2m_t.reshape(-1)
s2p = s2 - s2m
v1 = model.get_weights()[2]
b1 = model.get_weights()[3]    

b1_t = logit(s2m_t) - np.matmul(s1m, v1).reshape(-1)
J = np.diag(expit(b1_t) * (1 - expit(b1_t)))
w_mat = get_w_mat(Bxy_hat, Phi1, Psi2, J)

v1_t = w_mat.T

        
for subfig in range(3):
    plt.subplot('14' + str(subfig + 1))
    plt.plot(range(k1), v1[:, subfig], 'r', label="Training")
    plt.plot(range(k1), v1_t[:, subfig], 'b', label="Theoretical")
    plt.legend(loc='lower left')
    plt.title('w(' + str(subfig + 1) + ')')
plt.subplot('144')
plt.plot(range(k2), b1, 'r', label="Training")
plt.plot(range(k2), b1_t, 'b', label="Theoretical")
plt.legend(loc='lower left')
plt.title('c')
plt.draw()
plt.pause(.001)    
print("v1  :", v1)
print("v1_t:", v1_t)
print("b1  :", b1)
print("b1_t:", b1_t)
print("s2m:", s2m)
print("s2m_t:", s2m_t.T)
print("--- %s seconds ---" % (time.time() - start_time))
