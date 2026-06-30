import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Define your file path and chunk size
bin_file_path = '../data/selfplay2.bin'  # Path to your binary file
chunk_size = 25000

# 2. Define the binary struct layout matching your C++ PositionRecord.
# Standard alignment: float (4 bytes) + 4 bytes padding + 12 uint64s (96 bytes) = 104 bytes.
record_dtype = np.dtype([
    ('bitboards', 'u8', 12),
    ('confidence', 'f4'),
])

record_size = record_dtype.itemsize 
print(f"Expected size per record: {record_size} bytes")

# Initialize an empty list to store the averages for each bucket
bucket_averages = []

print("Processing binary file chunk by chunk...")

# 3. Stream the binary file in chunks
if not os.path.exists(bin_file_path):
    print(f"Error: File '{bin_file_path}' not found.")
    exit()

with open(bin_file_path, 'rb') as f:
    chunk_number = 1
    bytes_to_read = chunk_size * record_size
    
    while True:
        buffer = f.read(bytes_to_read)
        if not buffer:
            break  # End of file reached
            
        # Convert binary buffer into a numpy structured array
        # count=-1 automatically handles the final chunk if it's smaller than chunk_size
        data = np.frombuffer(buffer, dtype=record_dtype)
        
        # Calculate the mean of the 'confidence' column for the current chunk
        chunk_mean = data['confidence'].mean()
        bucket_averages.append(chunk_mean)
        
        # Print progress every 50 chunks
        if chunk_number % 50 == 0:
            print(f"Processed {chunk_number * chunk_size:,} rows...")
            
        chunk_number += 1

print("Processing complete! Generating plot...")

# 4. Plot the resulting data points
plt.figure(figsize=(10, 6))
plt.plot(bucket_averages, linestyle='-', marker='.', markersize=4, color='royalblue', alpha=0.8)

# Add descriptions and labels (Dynamic based on chunk_size variable)
plt.title(f'Average Position Confidence per {chunk_size:,} Plies (Binary Data)', fontsize=14)
plt.xlabel(f'Bucket Index (Each point represents {chunk_size:,} rows)', fontsize=12)
plt.ylabel('Average Confidence (0.0 to 1.0)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 5. Display the plot
plt.show()