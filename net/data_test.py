import pandas as pd
import matplotlib.pyplot as plt

# 1. Define your file path and chunk size
csv_file_path = '../data/selfplay_4_fixed3.csv'  # Replace with your actual file name
chunk_size = 25000

# 2. Initialize an empty list to store the averages for each bucket
bucket_averages = []

print("Processing file chunk by chunk...")

# 3. Stream the file in chunks of 100,000 rows
# 'header=None' tells pandas not to look for a header row
# 'names' explicitly assigns names to the columns so 'usecols' knows what to load
for chunk_number, chunk in enumerate(
    pd.read_csv(
        csv_file_path, 
        chunksize=chunk_size, 
        header=None, 
        names=['FEN_STRING', 'EVALUATION'], 
        usecols=['EVALUATION']
    ), 
    start=1
):
    
    # Calculate the mean of the 'EVALUATION' column for the current 100k rows
    chunk_mean = chunk['EVALUATION'].mean()
    bucket_averages.append(chunk_mean)
    
    # Print progress every 50 chunks (~5 million rows)
    if chunk_number % 50 == 0:
        print(f"Processed {chunk_number * chunk_size:,} rows...")

print("Processing complete! Generating plot...")

# 4. Plot the resulting 600 data points (60M / 100k)
plt.plot(bucket_averages, linestyle='-', marker='.', markersize=4, color='royalblue', alpha=0.8)

# Add descriptions and labels
plt.title('Average Position Evaluation per 100k Lines (No Header CSV)', fontsize=14)
plt.xlabel('Bucket Index (Each point represents 100,000 rows)', fontsize=12)
plt.ylabel('Average Evaluation (0.0 to 1.0)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 5. Display the plot
plt.show()