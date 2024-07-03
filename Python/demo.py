# Copyright (c) by Alejandro Ordonez Conejo under BSD License
# Last modified: Armin Lederer 02/2022

from LoGGP import LoGGP

import time
import numpy as np
import matplotlib.pyplot as plt

Ntrain = 100**2
dx = 2
dy = 1
ard = True
Nbar = 100
Nlocmod = 100000

X = np.mgrid[-5:5.1:complex(0, np.sqrt(Ntrain)), -5:5.1:complex(0, np.sqrt(Ntrain))].reshape(2, -1)
Y = np.sin(X[0, :]) + np.cos(X[1, :])

perm = np.random.permutation(Y.shape[0])
X = X[:, perm-1]
Y = Y[perm-1]

gp = LoGGP(dx, dy, Nbar, Nlocmod, ard)
gp.wo = 100

# Online hyperparameter optimization is currently not supported.
# For optimizing hyperparameters, software such as GPytorch can be used.
gp.sigmaF = 1.0*np.ones((1, 1))
gp.sigmaN = 0.1*np.ones((1, 1))
gp.lengthS = 1.0*np.ones((dx, 1))

predtime = np.zeros((Ntrain, 1))
uptime = np.zeros((Ntrain, 1))
smse = np.zeros((Ntrain, 1))

for i in range(Ntrain):
    t1 = time.time_ns()
    output = gp.predict(X[:, i])
    predtime[i] = time.time_ns() - t1
    
    t1 = time.time_ns()
    gp.update(X[:, i], Y[i]+np.random.normal(0.0, gp.sigmaN[0], dy))
    uptime[i] = time.time_ns() - t1
    
    smse[i] = ((output - Y[i]) ** 2) / Y.var()

cum_smse = np.cumsum(smse)/(np.linspace(1, Ntrain, Ntrain))

plt.semilogy(cum_smse)
plt.title('cumulative standardized mean squared error')
plt.show()

plt.semilogy(uptime/(10**9))
plt.title('update time[s]')
plt.show()

plt.semilogy(predtime/(10**9))
plt.title('prediction time[s]')
plt.show()
