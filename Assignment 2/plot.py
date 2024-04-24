import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data sizes and configurations
data_sizes = [4096**2, 8192**2]  # Square of 4096 and 8192
stencil_configs = ['With Leader', 'Without Leader']

# Define the actual time values as given
actual_time_values = {
    (8, 4096**2, 'With Leader'):    [5.062379 ,4.439462  ,4.577898 ,4.097052 ,4.105396 ],
    (8, 4096**2, 'Without Leader'): [4.581456 ,4.186064  ,4.514985 ,4.805784 ,4.757033 ],  
    (8, 8192**2, 'With Leader'):    [5.657943 ,5.672132  ,5.729710 ,6.258561 ,5.360255 ],
    (8, 8192**2, 'Without Leader'): [5.753336 ,5.635435  ,5.901573 ,6.787521 ,5.879524 ],
    (12, 4096**2, 'With Leader'):   [20.996001,19.998650 ,20.183046,19.244533,20.127660],
    (12, 4096**2, 'Without Leader'):[20.954927,20.315492 ,20.133259,19.676669,20.654986],  
    (12, 8192**2, 'With Leader'):   [21.845758,22.004273 ,22.629298,21.382664,23.051134],
    (12, 8192**2, 'Without Leader'):[22.105198,22.463110 ,21.218704,21.985296,23.055752],
}

# Prepare data for DataFrame
data = []
for num_processes in [8, 12]:
    for size in data_sizes:
        for config in stencil_configs:
            for time_taken in actual_time_values.get((num_processes, size, config), []):
                data.append([num_processes, size, config, time_taken])

# Create DataFrame
df = pd.DataFrame(data, columns=['Num Processes', 'Data Size', 'Stencil Configuration', 'Time (s)'])
df['Processes, Size, Config'] = df.apply(
    lambda row: f"{row['Num Processes']}P, {int(np.sqrt(row['Data Size']))}^2, {row['Stencil Configuration']}", axis=1
)

# Plotting
plt.figure(figsize=(14, 8))
sns.boxplot(x='Processes, Size, Config', y='Time (s)', data=df)
plt.title('Time Taken by Main Halo Exchange Function for Various Data Sizes with and without leader')
plt.xticks(rotation=45)
plt.ylabel('Time (seconds)')
plt.xlabel('(No. of Processes, Data Size, Leader/No Leader)')
plt.grid(True)
plt.tight_layout()

# Change the path according to your location
plt.savefig('D:/MPI Ubuntu/CS633/Group01/plot.png')
plt.show()
