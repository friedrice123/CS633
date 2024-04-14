import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data sizes and configurations
data_sizes = [4096**2, 8192**2]  # Square of 4096 and 8192
stencil_configs = ['With Leader', 'Without Leader']

# Define the actual time values as given
actual_time_values = {
    (8, 4096**2, 'With Leader'): [2.108471,4.241110,3.976440,4.097052,4.105396],
    (8, 4096**2, 'Without Leader'): [1.851123,3.721280,4.396117,4.805784,3.757033],  
    (8, 8192**2, 'With Leader'): [4.664802,6.150822,4.828390,6.258561,6.360255],
    (8, 8192**2, 'Without Leader'): [3.920070,7.790995,5.462743,5.787521,5.879524],
    (12, 4096**2, 'With Leader'): [19.425452, 21.507802,15.046802,23.244533,17.127660],
    (12, 4096**2, 'Without Leader'): [14.803349,17.507736,15.031597,17.676669,14.654986],  
    (12, 8192**2, 'With Leader'): [28.229459,30.962640,18.630124,27.182664,24.051134],
    (12, 8192**2, 'Without Leader'): [22.547943,29.899688,14.769674,24.985296,23.055752],
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
