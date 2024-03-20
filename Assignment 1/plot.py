import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data sizes (squared values) and stencil configurations
data_sizes = [512**2, 2048**2]  # Data sizes
stencil_configs = ['5', '9']  # Stencil configurations

# Placeholder for actual time values
actual_time_values = {
    (512**2, '5'): [0.051235, 0.04091, 0.052509],  # Example values; replace with your actual data
    (512**2, '9'): [0.059266, 0.074726, 0.066472],
    (2048**2, '5'): [0.463102, 0.555919, 0.3548],
    (2048**2, '9'): [0.697287, 0.594133, 0.627878],
}

# Generate a DataFrame with the actual execution times
data = []
for size in data_sizes:
    for config in stencil_configs:
        for time_taken in actual_time_values.get((size, config), []):
            data.append([size, config, time_taken])

df = pd.DataFrame(data, columns=['Data Size', 'Stencil Configuration', 'Time (s)'])
df['Size, Config'] = df.apply(lambda row: f"{int(np.sqrt(row['Data Size']))}^2, {row['Stencil Configuration']}", axis=1)

# First plot: Original times
plt.figure(figsize=(14, 8))
sns.boxplot(x='Size, Config', y='Time (s)', data=df)
plt.title('Time Taken by Main Halo Exchange Function for Various Data Sizes and Stencil Configurations')
plt.xticks(rotation=45)
plt.ylabel('Time (seconds)')
plt.xlabel('(Data Size, Stencil Configuration)')
plt.grid(True)
plt.tight_layout()
plt.savefig('E:\Sem6\CS633\CS633\Assignment 1\plot.png')  # Save the first plot
