import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
import numpy as np
import random

# Global lookups to eliminate recreation overhead inside the 74M loop
PIECE_MAP_W = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
PIECE_MAP_B = {'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11, 'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
TARGET_MAP = {"-1": 0.0, "0": 0.5, "1": 1.0, "-1.0": 0.0, "0.0": 0.5, "1.0": 1.0}

def fen_to_768(fen: str) -> torch.Tensor:
    features = np.zeros(768, dtype=np.float32)
    
    try:
        parts = fen.split(' ', 2)
        placement, side_to_move = parts[0], parts[1]
        
        is_white = (side_to_move == "w")
        piece_map = PIECE_MAP_W if is_white else PIECE_MAP_B
        
        square = 0
        for char in placement:
            if char == '/':
                continue
            if '1' <= char <= '8':
                square += ord(char) - 48
            else:
                actual_square = square if is_white else (square ^ 56)
                features[piece_map[char] * 64 + actual_square] = 1.0
                square += 1
                
    except Exception:
        pass
        
    return torch.from_numpy(features)

class ChessStreamingDataset(IterableDataset):
    def __init__(self, csv_file, buffer_size=100000):
        self.csv_file = csv_file
        self.buffer_size = buffer_size
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Could not find the data file: {csv_file}")

    def _process_line(self, line):
        parts = line.rsplit(',', 1)
        if len(parts) < 2:
            return None
        
        fen = parts[0]
        target = float(parts[1].strip())
        
        feature_vector = fen_to_768(fen)
        return (
            feature_vector,
            torch.tensor([target], dtype=torch.float32)
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        with open(self.csv_file, 'r', encoding='utf-8') as f:
            buffer = []
            for line_idx, line in enumerate(f):
                if line_idx % num_workers != worker_id:
                    continue
                    
                if not line.strip():
                    continue
                
                if len(buffer) < self.buffer_size:
                    buffer.append(line)
                else:
                    idx = random.randint(0, self.buffer_size - 1)
                    processed = self._process_line(buffer[idx])
                    buffer[idx] = line 
                    
                    if processed is not None:
                        yield processed
            
            random.shuffle(buffer)
            for line in buffer:
                processed = self._process_line(line)
                if processed is not None:
                    yield processed


# ==========================================
# 3. Neural Network Architecture
# ==========================================
class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()
        self.hidden_layer = nn.Linear(768, 64)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)
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

    csv_filename = "../data/selfplay_4.csv" 
    batch_size = 4096*2*2
    learning_rate = 0.001
    epochs = 5

    dataset = ChessStreamingDataset(csv_filename, buffer_size=50000)

    num_workers = 8
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=1
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
                h.write(f"{loss.item()}\n")
                progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", loss=f"{loss.item():.5f}")

    progress_bar.close()

    model_save_path = "64hl1.pt"
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(orig_model.state_dict(), model_save_path)
    print(f"Training completed. Network saved safely to {model_save_path}")

if __name__ == "__main__":
    main()