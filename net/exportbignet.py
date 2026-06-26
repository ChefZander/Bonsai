import os
import torch
import numpy as np

def export_to_hpp(model_path="chess_value_net.pt", output_path="chess_net.hpp", scale=64):
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please ensure training is complete.")
        return

    print(f"Loading weights from {model_path}...")
    # Load state dict safely to CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract weights as numpy arrays from the new architecture
    w0 = state_dict['hl0.weight'].numpy()  # Shape: (256, 768)
    b0 = state_dict['hl0.bias'].numpy()    # Shape: (256,)
    
    w1 = state_dict['hl1.weight'].numpy()  # Shape: (256, 256)
    b1 = state_dict['hl1.bias'].numpy()    # Shape: (256,)
    
    w2 = state_dict['hl2.weight'].numpy()  # Shape: (256, 256)
    b2 = state_dict['hl2.bias'].numpy()    # Shape: (256,)
    
    w3 = state_dict['hl3.weight'].numpy()  # Shape: (1, 256)
    b3 = state_dict['hl3.bias'].numpy()    # Shape: (1,)

    # Quantize weights to integers (always scaled by base factor)
    w0_int = np.round(w0 * scale).astype(int)
    w1_int = np.round(w1 * scale).astype(int)
    w2_int = np.round(w2 * scale).astype(int)
    w3_int = np.round(w3 * scale).astype(int)

    # Accumulate scales for biases based on deep layer progression
    b0_int = np.round(b0 * scale).astype(int)
    b1_int = np.round(b1 * (scale ** 2)).astype(int)
    b2_int = np.round(b2 * (scale ** 3)).astype(int)
    b3_int = np.round(b3 * (scale ** 4)).astype(int)

    # Helper function to stringify 1D array
    def format_1d(arr):
        return ", ".join(map(str, arr))

    # Helper function to stringify 2D array
    def format_2d(arr):
        rows = []
        for row in arr:
            rows.append("    {" + ", ".join(map(str, row)) + "}")
        return ",\n".join(rows)

    # Generate the C++ HPP template matching the 256-node hidden layers
    hpp_content = f"""#ifndef CHESS_NET_HPP
#define CHESS_NET_HPP

#include <cstdint>

// Quantization Parameters
const int SCALE_FACTOR = {scale};

// Layer 0 (Input to Hidden 1): Shape [256][768]
const int16_t HL0_WEIGHTS[256][768] = {{
{format_2d(w0_int)}
}};
const int16_t HL0_BIASES[256] = {{
    {format_1d(b0_int)}
}};

// Layer 1 (Hidden 1 to Hidden 2): Shape [256][256]
const int16_t HL1_WEIGHTS[256][256] = {{
{format_2d(w1_int)}
}};
const int32_t HL1_BIASES[256] = {{
    {format_1d(b1_int)}
}};

// Layer 2 (Hidden 2 to Hidden 3): Shape [256][256]
const int16_t HL2_WEIGHTS[256][256] = {{
{format_2d(w2_int)}
}};
const int32_t HL2_BIASES[256] = {{
    {format_1d(b2_int)}
}};

// Layer 3 (Output Layer): Shape [256] (Flattened out_features x in_features)
const int16_t OUTPUT_WEIGHTS[256] = {{
    {format_1d(w3_int[0])}
}};
const int32_t OUTPUT_BIAS = {b3_int[0]};

#endif // CHESS_NET_HPP
"""

    with open(output_path, "w") as f:
        f.write(hpp_content)
        
    print(f"Successfully generated header file: {output_path}")
    print(f"Weights scaled by {scale}")
    print(f"Biases scaled progressly: HL0={scale}, HL1={scale**2}, HL2={scale**3}, Output={scale**4}")

if __name__ == "__main__":
    export_to_hpp(model_path="bignet2.pt", output_path="bignet2.hpp", scale=64)