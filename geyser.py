import pandas as pd
from scipy.stats import norm

# Corrected raw URL for the geyser dataset
url = 'https://raw.githubusercontent.com/sujaysavanth/statisticsfordata/main/geyser.txt'

# Load the data into a DataFrame
df = pd.read_csv(url, delimiter='\s+', index_col=False)

# Extract the 'waiting' column
data = df.waiting
n = len(data)

# Calculate the sample mean and standard error
theta = data.mean()
se = data.std()

# Z-value for a 90% confidence interval
z_90 = norm.ppf(0.95)

# Print the estimated mean and confidence interval
print("Estimated mean: %.3f , Standard error: %.3f" % (theta, se))
print("90%% interval: (%.3f , %.3f)" % (theta - z_90 * se, theta + z_90 * se))

# Calculate and print the median
median = data.median()
print("Estimated median time: %.3f" % median)
