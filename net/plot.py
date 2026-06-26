import matplotlib.pyplot as plt
import pandas as pd

# Specify the path to your data file
file_path = 'train_log.csv'

# Read the floats from the file
data = []
with open(file_path, 'r') as file:
    for line in file:
        cleaned_line = line.strip()
        if cleaned_line:  # Skip empty lines
            data.append(float(cleaned_line))

# Calculate the rolling average using pandas
# You can increase window_size for a smoother line, or decrease it for more detail
window_size = 100
rolling_avg = pd.Series(data).rolling(window=window_size).mean()

# Create the plot using subplots
fig, ax = plt.subplots()

# Plot the original data with alpha=0.4 to make it less visually dominant
ax.plot(data, label='Raw Data', linewidth=1, alpha=0.4)

# Plot the rolling average line
ax.plot(rolling_avg, label=f'{window_size}-Step Rolling Average', linewidth=2, color='firebrick')

# Add labels, title, and grid
ax.set_yscale("log")
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Plot of Floats from File with Rolling Average')
ax.grid(True)
ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.show()