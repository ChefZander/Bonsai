import os
import torch
import numpy as np

def export_to_hpp(model_path="chess_value_net.pt", output_path="chess_net.hpp", scale=64, hidden_layer=16):
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please ensure training is complete.")
        return

    print(f"Loading weights from {model_path}...")
    # Load state dict safely to CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract weights as numpy arrays
    w1 = state_dict['hidden_layer.weight'].numpy()  # Shape: (hidden_layer, 768)
    b1 = state_dict['hidden_layer.bias'].numpy()    # Shape: (hidden_layer,)
    w2 = state_dict['output_layer.weight'].numpy()  # Shape: (1, hidden_layer)
    b2 = state_dict['output_layer.bias'].numpy()    # Shape: (1,)

    # Quantize weights to integers
    w1_int = np.round(w1 * scale).astype(int)
    b1_int = np.round(b1 * scale).astype(int)
    w2_int = np.round(w2 * scale).astype(int)
    
    # CRITICAL: Since Layer 1 output is already scaled by 'scale', 
    # Layer 2's multiplication pushes the scale to 'scale * scale'.
    # Layer 2 bias must match this scale exactly.
    b2_int = np.round(b2 * scale * scale).astype(int)

    # Helper function to stringify 1D array
    def format_1d(arr):
        return ", ".join(map(str, arr))

    # Helper function to stringify 2D array
    def format_2d(arr):
        rows = []
        for row in arr:
            rows.append("    {" + ", ".join(map(str, row)) + "}")
        return ",\n".join(rows)

    # Generate the C++ HPP template
    hpp_content = f"""#ifndef CHESS_NET_HPP
#define CHESS_NET_HPP

#include <cstdint>

// Quantization Parameters
const int SCALE_FACTOR = {scale};
const int SCALE_FACTOR_SQ = {scale * scale};

// Hidden Layer Weights: Shape [{hidden_layer}][768] (out_features x in_features)
const int16_t HIDDEN_WEIGHTS[{hidden_layer}][768] = {{
{format_2d(w1_int)}
}};

// Hidden Layer Biases: Shape [{hidden_layer}]
const int16_t HIDDEN_BIASES[{hidden_layer}] = {{
    {format_1d(b1_int)}
}};

// Output Layer Weights: Shape [{hidden_layer}] (Flattened out_features x in_features)
const int16_t OUTPUT_WEIGHTS[{hidden_layer}] = {{
    {format_1d(w2_int[0])}
}};

// Output Layer Biases: Shape [1]
const int32_t OUTPUT_BIAS = {b2_int[0]};

#endif // CHESS_NET_HPP
"""

    with open(output_path, "w") as f:
        f.write(hpp_content)
        
    print(f"Successfully generated header file: {output_path}")
    print(f"Weights scaled by {scale}, Output Bias scaled by {scale * scale}")

if __name__ == "__main__":
    # You can change scale=1 if you strictly want raw non-scaled rounding
    export_to_hpp(model_path="24hl1.pt", output_path="24hl1.hpp", scale=64, hidden_layer=64)