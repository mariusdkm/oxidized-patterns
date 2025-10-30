import os
import sys
import random
import string
from tqdm import tqdm

def generate_large_file(filename="random_5gb.txt", size_gb=5):
    # Convert GB to bytes
    target_size = size_gb * 1024 * 1024 * 1024
    
    # Prepare a large string of possible characters
    characters = string.ascii_letters + string.digits + string.punctuation
    
    print(f"Generating {size_gb}GB file with random ASCII characters...")
    
    # Initialize progress bar for chunk generation
    chunk_pbar = tqdm(total=target_size, unit='B', unit_scale=True, desc="Generating chunks")
    
    lines = []
    total_size = 0
    line_length = 1
    
    # Generate chunks with progress tracking
    while total_size < target_size:
        actual_len = min(line_length, target_size - total_size)
        line = ''.join(random.choices(characters, k=int(actual_len - 1))) + '\n'
        line_size = len(line.encode())
        lines.append(line)
        total_size += line_size
        chunk_pbar.update(line_size)
        
        if line_length >= 16384:
            line_length = 0
        line_length += 1
    
    chunk_pbar.close()
    
    # shuffle the lines
    seed = 42
    random.seed(seed)
    random.shuffle(lines)
    
    # Write the file in chunks with progress bar
    print("Writing data to file...")
    with open(filename, 'w') as f:
        write_pbar = tqdm(lines, desc="Writing to disk")
        for line in write_pbar:
            f.write(line)
    
    # Get actual file size
    actual_size = os.path.getsize(filename)
    print(f"\nFile generated: {filename}")
    print(f"Target size: {target_size:,} bytes")
    print(f"Actual size: {actual_size:,} bytes")
    print(f"Difference: {abs(target_size - actual_size):,} bytes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_random_data.py <size>")
        sys.exit(1)
    
    size = float(sys.argv[1])
    generate_large_file(f"random_{size}gb.txt", size)
