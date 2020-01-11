import numpy as np
import networkx as nx
import scipy
from matplotlib import pyplot as plt

from scipy.stats import chisquare

import sys

from dppy.exotic_dpps_core import ust_sampler_wilson_nodes
from dppy.exotic_dpps import UST
from dppy.utils import (det_ST, example_eval_L_linear, example_eval_L_min_kern)

# Parameters
q = 1.0
nbr_it = 10000


# Generate graph
n, p = 5, 0.4
not_connected = True
while not_connected:
    G = nx.erdos_renyi_graph(n, p)
    if nx.is_connected(G):
        not_connected = False
ust = UST(G)
ust.plot_graph()

# Get laplacian and adjacency matrix
L = scipy.sparse.csr_matrix(nx.laplacian_matrix(G), dtype='d')
W = scipy.sparse.csr_matrix(nx.adjacency_matrix(G), dtype='d')

# Build K_q
A = L.toarray()
V, U = np.linalg.eigh(A)
g = q/(q+V)
gdiag = np.diag(g)
K_q = U.dot(gdiag).dot(U.transpose())

Samples = []
Singletons_count = np.zeros(n)
Pairs_count = np.zeros((n, n))
cardinals = []

# Generate samples and count the occurences of each node and each pair of nodes
for i in range(nbr_it):
    Y, all_path, _ = ust_sampler_wilson_nodes(W, absorbing_weight=q)
    Samples.append(Y)
    cardinals.append(len(Y))
    Singletons_count[Y] += 1
    if len(Y) > 1:
        pairs_Y = [(Y[k1], Y[k2]) for k1 in range(len(Y)) for k2 in range(k1+1, len(Y))]
        pairs_Y = np.array(pairs_Y)
        Pairs_count[pairs_Y[:, 0], pairs_Y[:, 1]] += 1
Pairs_count = Pairs_count + Pairs_count.T

# Compute the theoritical and empirical distribution of each node
singleton_marginal_th = np.diag(K_q) / np.trace(K_q)
singleton_marginal_th /= np.sum(singleton_marginal_th)
singleton_marginal_emp = Singletons_count / nbr_it
singleton_marginal_emp /= np.sum(singleton_marginal_emp)
_, pval_singleton = chisquare(f_obs=singleton_marginal_emp, f_exp=singleton_marginal_th)

# Compute the theoritical and empirical distribution of each pair of nodes
all_pairs = [(k1, k2) for k1 in range(n) for k2 in range(k1+1, n)]
all_pairs_array = np.array(all_pairs)
# det [[K_ii, K_ij], [K_ji, K_jj]]
pair_marginal_th = np.array([det_ST(K_q, list(d)) for d in all_pairs])
pair_marginal_emp = Pairs_count[all_pairs_array[:, 0], all_pairs_array[:, 1]].reshape(-1) / nbr_it
_, pval_pair = chisquare(f_obs=pair_marginal_emp, f_exp=pair_marginal_th)


print('------------------Cardinal------------------')
print('Theoretical =', np.sum(g))
print('Empirical =', np.mean(cardinals))
print('--------------------------------------------')
print()
print('-----------------Singletons-----------------')
print('Theoretical =',singleton_marginal_th)
print('Empirical =',singleton_marginal_emp)
print('p-value =', pval_singleton)
print('--------------------------------------------')
print()
print('-------------------Pairs--------------------')
print('Theoretical =',pair_marginal_th)
print('Empirical =',pair_marginal_emp)
print('p-value =', pval_pair)
print('--------------------------------------------')




    


