#################
#Name - Sayem Lincoln
#PID - A54207835
#Homework 8 Problem 5
#################

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

from scipy.io import loadmat

# Reading the data from the two mat files called "data.mat" and "USPS.mat"

data = loadmat('data.mat')
usps = loadmat('USPS.mat')

#################
# PART  I  :  (A)
#################

## First Data Set "data.mat"

X = pd.DataFrame(data['X'])
X.shape

X.head()

y = pd.DataFrame(data['Y'])
y.shape

y[0].unique()

# CVXOPT solver and resulting  α

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

#Initializing values and computing H. Note the 1. to force to float type
m,n = X.shape
y = y.values.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])


# Compute  w  and  b  parameters

#w parameter in vectorized form
w = ((y * alphas).T @ X.values).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing b
b = y[S] - np.dot(X[S], w)

#Display results
#print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten(), '\n')
print('b = ', b[0])

## Second Data Set "USPS.mat"

# We will do the same thing, extract the data and then calculate CVXOPT solver for this data set.

X = pd.DataFrame(usps['A'])
X.shape

X.head()

y = pd.DataFrame(usps['L'])
y.shape

y[0].unique()

# In this case, we have a multiclass problem not like the previous one (Binary classification)

# CVXOPT solver and resulting  α

#Initializing values and computing H. Note the 1. to force to float type
m,n = X.shape
y = y.values.reshape(-1,1) * 1.
X_dash = y * X
X_dash = preprocessing.scale(X_dash)
H = np.dot(X_dash , X_dash.T) * 1.


#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

# Compute  w  and  b  parameters

#w parameter in vectorized form
w = ((y * alphas).T @ X.values).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing b
b = y[S] - np.dot(X[S], w)

#Display results
#print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten(), '\n')
print('b = ', b)

##################
# PART  II  :  (B)
##################



# Now we investigate the second part of the question : Using random datasets (X, y) as input to the algorithm and vary the sample size and feature dimensionality.

def solve_random_qp(m, n):
    X, y = np.random.random((m, n)), random.sample(([1]*m+[-1]*m), m)
    y = np.array(y).reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    
    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    cvxopt_solvers.options['show_progress'] = False
    
    start = time.time()
    cvxopt_solvers.qp(P, q, G, h, A, b)
    
    return (time.time() - start)

## Varying the Sample size:

m_sizes = [50, 100, 200, 500, 1000, 2000, 3000, 4000]

m_times = []
for size in m_sizes:
    print("Running on sample size %d..." %  size)
    m_times.append(solve_random_qp(size, 5))

plt.plot(m_sizes, m_times, lw=2, marker='o')
plt.grid(True)
plt.xscale('log')
plt.ylabel('Time (s)');
plt.xlabel('Sample size m');
plt.title('Time costs of the sample sizes');

## Varying the Feature size:

n_sizes = [5, 10, 20, 50, 100, 300, 500]

n_times = []
for size in n_sizes:
    print("Running on feature size %d..." %  size)
    n_times.append(solve_random_qp(1000, size))


plt.plot(n_sizes, n_times, lw=2, marker='o')
plt.grid(True)
plt.xscale('log')
plt.ylabel('Time (s)');
plt.xlabel('Feature size n');
plt.title('Time costs of the feature sizes');


# We can remark that varying number of samples is more time consuming than varying the number of features for this solver.
