"""
First Experiment: Feature Projection in Neural Networks
[Sec 3.4.1]
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from scipy.special import expit # calculate Logistic sigmoid

import scipy.io as scio
from func import MakeLabels, regulate, p2b, GenerateDiscreteSamples, get_p_hat_mat, print_for_tikz
plt.ion()


def get_trained_vaules(weights):
    "compute f, g from trained weights"
    s = weights[0].reshape(1, xCard)
    s = expit(s + weights[1])
    f, _ = regulate(s, Px, axis = 1)
    f = f * np.sign(sum(f*f_theory)) # to flip the sign (if needed)
    v = weights[2]
    g, _ = regulate(v, Py, axis = 1) 
    g = g * np.sign(sum(g*g_theory)) # to flip the sign (if needed)
    mu = np.matmul(s, Px)
    bias = weights[3]
    b, _ = regulate(bias, Py, unilen = False) # Normalize to have zero mean
    b_theory = np.log(Py) - mu * v # note that b_theory depends on training results
    b_theory = b_theory.reshape(-1)
    b_theory, _ = regulate(b_theory, Py, unilen = False) # Normalize to have zero mean
    return f, g, b, b_theory

# train a neural network with ideal expressive power
def train_weights(XLabels, YLabels):
    nSamples = XLabels.shape[0]
    xCard = XLabels.shape[1]
    yCard = YLabels.shape[1]
    
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=xCard))
    model.add(Dense(yCard, activation='softmax', input_dim=1))

    sgd = SGD(4, decay=1e-2, momentum=0.9, nesterov=True, clipnorm=.5)

    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    model.fit(XLabels, YLabels, verbose=0, batch_size=nSamples, epochs=200) #batch_size=50000
    weights = model.get_weights()
    return weights


xCard = 8 # cardinalities of X
yCard = 6 # cardinalities of Y

nSamples = 100000 # number of samples

np.random.seed(1)
# randomly pick joint distribution, normalize
Pxy = np.random.random([yCard, xCard])
Pxy = Pxy / sum(sum(Pxy))

# compute marginals
Px = np.sum(Pxy, axis = 0)
Py = np.sum(Pxy, axis = 1)    

[X, Y] = GenerateDiscreteSamples(Pxy, nSamples)

XLabels = MakeLabels(X)
YLabels = MakeLabels(Y)


'''
Compute theoretical answers for f and g, corresponding to the 1st pair
of singular vectors of B
'''
# or just use Pxy to approximate Pxy_hat
Pxy_hat = get_p_hat_mat(XLabels, YLabels)
Px_hat = np.sum(Pxy_hat, axis = 0)
Py_hat = np.sum(Pxy_hat, axis = 1)    

B = p2b(Pxy_hat)
U, s, V = np.linalg.svd(B)
f_theory = V[1,:] / np.sqrt(Px_hat)
g_theory = U[:,1] / np.sqrt(Py_hat)



weights_list = []

cnt = 1  # may change to larger values (e.g., 10) to see the standard deviation
for i in range(cnt):
# allow random training
    np.random.seed(None)
    weights = train_weights(XLabels, YLabels)
    weights_list += [weights]


# read data from 10 repeated experiments
f_mat = np.zeros([cnt, xCard])
g_mat = np.zeros([cnt, yCard])
for i in range(cnt):
    f_trained, g_trained, _, _ = get_trained_vaules(weights_list[i])
    f_mat[i] = f_trained
    g_mat[i] = g_trained

f_mean = np.mean(f_mat, axis = 0)
g_mean = np.mean(g_mat, axis = 0)
f_std = np.std(f_mat, axis = 0)    
g_std = np.std(g_mat, axis = 0)


# compute results from the first weights
f, g, b, b_theory = get_trained_vaules(weights_list[0])

plt.figure(figsize = (9, 3))
plt.subplot(131)
plt.plot(range(xCard), f.reshape(-1),  'r', label='Training Result')
plt.plot(range(xCard), f_theory, 'b', label='Theoretic')
plt.legend(loc='lower left')
plt.title('Feature s')
plt.subplot(132)
plt.plot(range(yCard), g.reshape(-1), 'r', label='Training')
plt.plot(range(yCard), g_theory, 'b', label='Theoretical')
plt.legend(loc='lower left')
plt.title('Weight v')
plt.subplot(133)
# plt.plot(range(yCard), muvb_tilde, 'r', label='Training')
# plt.plot(range(yCard), logPy_tilde, 'b', label='Theoretical')
plt.plot(range(yCard), b, 'r', label='Training')
plt.plot(range(yCard), b_theory, 'b', label='Theoretical')
plt.legend(loc='lower left')
plt.title('Bias b')
plt.show()

print(print_for_tikz(f))
print(print_for_tikz(f_theory))
print_for_tikz(g)
print_for_tikz(g_theory)
print_for_tikz(b)
print_for_tikz(b_theory)
