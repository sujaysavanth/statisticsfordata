import pandas as pd
from scipy.stats import norm

# Corrected raw URL for the cloud dataset
url = 'https://raw.githubusercontent.com/sujaysavanth/statisticsfordata/main/cloud.txt'

# Load the data into a DataFrame
df = pd.read_csv(url, delimiter='\s+', index_col=False)

# Extract the 'Seeded_Clouds' and 'Unseeded_Clouds' columns
seed_data = df.Seeded_Clouds
no_seed_data = df.Unseeded_Clouds
n = len(seed_data)

# Calculate the means for the two groups
mean_seed = seed_data.mean()
mean_noseed = no_seed_data.mean()
theta_hat = mean_seed - mean_noseed

# Standard errors for the two groups
se_hat_seed = seed_data.std()/n**0.5 
se_hat_no_seed = no_seed_data.std()/n**0.5
se_hat = (se_hat_seed**2 + se_hat_no_seed**2)**0.5

# Z-value for 95% confidence interval
z_95 = norm.ppf(0.975)

# Print results
print("Estimated difference: %.3f" % theta_hat)
print("Standard error: %.3f" % se_hat)
print("95%% confidence interval: (%.3f, %.3f)" % (theta_hat - z_95 * se_hat, theta_hat + z_95 * se_hat))
