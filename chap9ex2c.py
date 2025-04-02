# Chapter 9, problem 2, part d
import numpy as np

a = 1.
b = 3.
n = 10
n_sims = 10**6

samples = np.random.uniform(low=a, high=b, size=[n_sims, n])
a_mle = np.min(samples, axis=1)
b_mle = np.max(samples, axis=1)

tau_mle = (a_mle + b_mle)/2 
tau = (a + b)/2
mse = np.mean((tau_mle - tau)**2)
print('Simulated MSE: %.3f' %mse)
