# Chapter 8, problem 2
import numpy as np
from scipy.stats import norm

n = 50
y = np.random.normal(0, 1, n)
x = np.exp(y)

def calc_skew(data):
    n, mu, sigma = data.size, data.mean(), data.std()
    tmp = 0;
    for element in data: tmp += (element - mu)**3
    return tmp/(n * sigma**3)

t_hat = calc_skew(x)

t_boot = []
B = 10000
for _ in range(B):
  sel_data = np.random.choice(x, size=n, replace=True)
  t_boot.append(calc_skew(sel_data))

se_hat = np.std(t_boot)
print('bootstrap standard error: %.4f' %se_hat)

z_95 = norm.ppf(.975)
print('95%% interval Normal method: (%.2f, %.2f)' %(t_hat-z_95*se_hat, t_hat+z_95*se_hat))

boot_quant_1 = np.quantile(t_boot, .05/2)
boot_quant_2 = np.quantile(t_boot, 1-.05/2)
print('95%% interval Pivotal method: (%.2f, %.2f)' %(2*t_hat-boot_quant_2, 2*t_hat-boot_quant_1))
print('95%% interval Percentile method: (%.2f, %.2f)' %(boot_quant_1, boot_quant_2))
