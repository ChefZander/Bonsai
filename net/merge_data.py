import glob
import os
import sys
import time

# --- CONFIGURATION ---
INPUT_PATTERN = "data/selfplay_*.bin"  # Matches all your worker files
OUTPUT_FILE = "data/combined.bin"       # The final merged file
CHUNK_SIZE = 16 * 1024 * 1024          # 16 MB buffer size for fast I/O
# ---------------------

def merge_binary_files():
    # Find all files matching the pattern
    input_files = glob.glob(INPUT_PATTERN)
    
    # Absolute paths to prevent accidentally reading the output file if it matches the pattern
    output_abs = os.path.abspath(OUTPUT_FILE)
    input_files = [os.path.abspath(f) for f in input_files if os.path.abspath(f) != output_abs]

    if not input_files:
        print(f"Error: No files found matching pattern '{INPUT_PATTERN}'")
        return

    print(f"Found {len(input_files)} files to merge.")
    print(f"Target output: {OUTPUT_FILE}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    start_time = time.time()
    total_bytes_written = 0

    # Open the output file in write-binary mode ('wb')
    # This will overwrite any existing combined.bin from previous runs
    with open(output_abs, 'wb') as outfile:
        for idx, file_path in enumerate(input_files, start=1):
            file_size = os.path.getsize(file_path)
            print(f"[{idx}/{len(input_files)}] Merging {os.path.basename(file_path)} ({file_size / (1024*1024):.2f} MB)...")
            
            with open(file_path, 'rb') as infile:
                while True:
                    chunk = infile.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    outfile.write(chunk)
                    total_bytes_written += len(chunk)

    end_time = time.time()
    elapsed = end_time - start_time
    total_mb = total_bytes_written / (1024 * 1024)
    
    print("\n--- MERGE COMPLETE ---")
    print(f"Successfully merged into: {OUTPUT_FILE}")
    print(f"Total data processed: {total_mb:.2f} MB")
    print(f"Time taken: {elapsed:.2f} seconds ({total_mb / elapsed:.2f} MB/s)")

if __name__ == "__main__":
    merge_binary_files()