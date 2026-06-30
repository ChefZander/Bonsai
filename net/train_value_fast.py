import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import numpy as np

class ChessStreamingDataset(IterableDataset):
    def __init__(self, bin_file, batch_size, buffer_size=200000):
        self.bin_file = bin_file
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Could not find the data file: {bin_file}")

        # 100-byte structural layout
        self.dtype = np.dtype([
            ('bitboards', np.uint64, 12),
            ('confidence', np.float32)
        ])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Open memmap inside the worker process
        data = np.memmap(self.bin_file, dtype=self.dtype, mode='r')
        total_records = len(data)

        # To avoid worker overlap, each worker owns a unique large block of the file
        block_size = self.buffer_size
        for block_start in range(worker_id * block_size, total_records, num_workers * block_size):
            block_end = min(block_start + block_size, total_records)
            if block_start >= total_records:
                break
            
            # Pull the entire block into RAM at once (Fast sequential OS read)
            chunk = np.array(data[block_start:block_end])
            if len(chunk) < self.batch_size:
                continue
                
            # Blazing-fast C-level shuffle of the entire block
            np.random.shuffle(chunk)
            
            # Slice the block into fully formed mini-batches
            for k in range(0, len(chunk), self.batch_size):
                chunk_slice = chunk[k:k + self.batch_size]
                
                # Drop trailing incomplete batches to maintain stable tensor shapes
                if len(chunk_slice) < self.batch_size:
                    continue 
                
                # VECTORIZED UNPACK: Unpack all 8,192 items simultaneously
                # View 12 uint64s as 96 raw bytes, then blast them into bits in one operation
                bytes_arr = chunk_slice['bitboards'].view(np.uint8).reshape(self.batch_size, 96)
                features_np = np.unpackbits(bytes_arr, axis=1, bitorder='little').astype(np.float32)
                targets_np = chunk_slice['confidence'].astype(np.float32).reshape(self.batch_size, 1)
                
                # Zero Python loops. Direct conversion to PyTorch tensors.
                yield torch.from_numpy(features_np), torch.from_numpy(targets_np)


# ==========================================
# 3. Neural Network Architecture
# ==========================================
class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()
        self.hidden_layer = nn.Linear(768, 32)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    torch.set_float32_matmul_precision('high')

    bin_filename = "../data/selfplay.bin" 
    batch_size = 4096 * 2  # 8192 (Left untouched)
    learning_rate = 0.001  # (Left untouched)
    epochs = 15            # (Left untouched)

    # Pass the batch size directly to the dataset
    dataset = ChessStreamingDataset(bin_filename, batch_size=batch_size, buffer_size=250000)

    num_workers = 8
    train_loader = DataLoader(
        dataset, 
        batch_size=None, # CRITICAL: Tells PyTorch to stop its slow item-by-item collation loop
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    model = ChessValueNet().to(device)

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled via torch.compile.")
        except Exception:
            pass

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print("Beginning training loop...")
    
    total_steps = epochs
    progress_bar = tqdm(total=total_steps, desc="Training Steps", unit="step")

    with open("train_log.csv", "w") as h:
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()

                progress_bar.update(1)
                if batch_idx % 8 == 0:
                    h.write(f"{loss.item()}\n")
                    progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", loss=f"{loss.item():.5f}")

    progress_bar.close()

    model_save_path = "32hl1.pt"
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(orig_model.state_dict(), model_save_path)
    print(f"Training completed. Network saved safely to {model_save_path}")

if __name__ == "__main__":
    main()