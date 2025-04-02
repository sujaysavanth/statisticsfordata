import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Use raw URL from GitHub for the file
url = "https://raw.githubusercontent.com/sujaysavanth/statisticsfordata/main/fiji.txt"

# Load the data into a DataFrame
df = pd.read_csv(url, delimiter='\s+', index_col=False)

# Function to calculate the ECDF and its bounds
def ecdf(data, x_vec, alpha):
    data_sort = np.sort(data)
    n = data.size
    eps = (np.log(2/alpha) / n / 2) ** 0.5
    F_hat, F_lower, F_upper = [], [], []
    
    for e in x_vec:
        if e >= data_sort[-1]:
            tmp = 1
        elif e < data_sort[0]:
            tmp = 0
        else:
            idx = np.argwhere(np.array(data_sort) > e)[0][0]
            tmp = idx / n
        F_hat.append(tmp)
        F_lower.append(max(tmp - eps, 0))
        F_upper.append(min(tmp + eps, 1))
    
    return F_hat, F_lower, F_upper

# Generate x values for the ECDF plot
x_vec = np.linspace(3.5, 7, 100)
data = df.mag  # Assuming 'mag' is the column of interest

# Compute the ECDF and its confidence intervals
F_hat, F_lower, F_upper = ecdf(data, x_vec, alpha=0.05)

# Plotting the ECDF and the confidence bounds
plt.plot(x_vec, F_hat, 'r.', lw=5, alpha=0.6, label='CDF empirical')
plt.plot(x_vec, F_lower, 'g.', lw=3, alpha=0.6, label='Lower bound')
plt.plot(x_vec, F_upper, 'b.', lw=3, alpha=0.6, label='Upper bound')
plt.legend(loc='best')
plt.show()

# Function to calculate the empirical CDF at a specific value
def calc_F_hat(data, x_val):
    data_sort = np.sort(data)
    if x_val >= data_sort[-1]:
        return 1
    elif x_val < data_sort[0]:
        return 0
    else:
        idx = np.argwhere(np.array(data_sort) > x_val)[0][0]
        return (idx / data.size)

# Calculate the difference between the CDF values at two points
a, b = 4.3, 4.9
theta_hat = calc_F_hat(data, b) - calc_F_hat(data, a)

# Standard error for the empirical CDF difference
se = (theta_hat * (1 - theta_hat) / data.size) ** 0.5

# 95% confidence interval for the difference in CDFs
z_95 = norm.ppf(0.975)
print('95% interval: ', round(theta_hat - z_95 * se, 3), ',', round(theta_hat + z_95 * se, 3))
