import torch
import torch.nn as nn
import numpy as np

def fen_to_768(fen: str) -> list:
    features = [0.0] * 768
    try:
        parts = fen.split()
        placement, side_to_move = parts[0], parts[1]
        
        if side_to_move == "w":
            piece_map = {
                'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
            }
        elif side_to_move == "b":
            piece_map = {
                'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
                'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5
            }
        
        square = 0
        for char in placement:
            if char == '/': continue
            elif char.isdigit(): square += int(char)
            else:
                piece_idx = piece_map[char]
                features[piece_idx * 64 + square] = 1.0
                square += 1
    except Exception:
        pass
    return features

class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()
        self.hidden_layer = nn.Linear(768, 16)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def main():
    # 1. Initialize and load model
    model = ChessValueNet()
    model_path = "chess_value_net.pt"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded weights successfully from {model_path}")
    except FileNotFoundError:
        print(f"Warning: '{model_path}' not found. Using initialized dummy weights for structural verification.")

    model.eval()

    # 2. Test Positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting line
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", # 1. e4
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # Open game
        "8/8/8/4k3/8/8/4K3/8 w - - 0 1"                               # King endgame
    ]

    print("\n--- PyTorch Inference Baseline Results ---")
    with torch.no_grad():
        for i, fen in enumerate(test_positions):
            features = fen_to_768(fen)
            input_tensor = torch.tensor([features], dtype=torch.float32)
            prediction = model(input_tensor).item()
            print(f"Pos #{i}: {prediction:.6f} | FEN: {fen}")
            

if __name__ == "__main__":
    main()