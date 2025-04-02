# Chapter 9, problem 3, part c
import numpy as np
from scipy.stats import norm

data = np.array([ 3.23, -2.50,  1.88, -0.68,  4.43,  0.17,
                  1.03, -0.07, -0.01,  0.76,  1.76,  3.18,
                  0.33, -0.31,  0.30, -0.61,  1.52,  5.43,
                  1.54,  2.28,  0.42,  2.33, -1.03,  4.00,
                  0.39])

# calc se using delta method
n = data.size
c = norm.ppf(0.95)
se_delta_method =  np.std(data) * np.sqrt((1 + 0.5*c**2)/n)
print('se delta method: %.3f' %se_delta_method)

# calc se using parametric bootstrap
n_sim = 10**5
samples = np.std(data) * np.random.randn(n_sims, n) + np.mean(data)
tau_boot = c * np.std(samples, axis=1) + np.mean(samples, axis=1)
print('se param bootstrap: %.3f' %np.std(tau_boot))
