import os
import torch
import numpy as np

def export_to_hpp(model_path="chess_value_net.pt", output_path="net4.hpp", scale=64):
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please ensure training is complete.")
        return

    print(f"Loading weights from {model_path}...")
    # Load state dict safely to CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract CNN and FC weights based on the new architecture
    w_conv = state_dict['conv.weight'].numpy()  # Shape: (1, 12, 3, 3)
    b_conv = state_dict['conv.bias'].numpy()    # Shape: (1,)
    w_fc = state_dict['fc.weight'].numpy()      # Shape: (1, 64)
    b_fc = state_dict['fc.bias'].numpy()        # Shape: (1,)

    # Quantize weights to integers
    w_conv_int = np.round(w_conv * scale).astype(int)
    b_conv_int = np.round(b_conv * scale).astype(int)
    w_fc_int = np.round(w_fc * scale).astype(int)
    
    # CRITICAL: Conv layer output is scaled by 'scale'. 
    # FC layer multiplication pushes the total scale to 'scale * scale'.
    # FC bias must match this scale exactly.
    b_fc_int = np.round(b_fc * scale * scale).astype(int)

    # Helper function to stringify 1D array
    def format_1d(arr):
        return ", ".join(map(str, arr))

    # Helper function to stringify 3D array [12][3][3] for the single out_channel
    def format_3d(arr_3d):
        channels = []
        for channel in arr_3d:
            rows = []
            for row in channel:
                rows.append("        {" + ", ".join(map(str, row)) + "}")
            channels.append("    {\n" + ",\n".join(rows) + "\n    }")
        return ",\n".join(channels)

    # Generate the C++ HPP template
    hpp_content = f"""#ifndef CHESS_NET_HPP
#define CHESS_NET_HPP

#include <cstdint>

// Quantization Parameters
const int SCALE_FACTOR = {scale};
const int SCALE_FACTOR_SQ = {scale * scale};

// Convolutional Weights: Shape [12][3][3] (in_channels x kernel_y x kernel_x)
// Output channels = 1, so the outermost dimension is omitted.
const int16_t CONV_WEIGHTS[12][3][3] = {{
{format_3d(w_conv_int[0])}
}};

// Convolutional Bias
const int32_t CONV_BIAS = {b_conv_int[0]};

// Fully Connected Layer Weights: Shape [64] (Flattened 8x8 input)
const int16_t FC_WEIGHTS[64] = {{
    {format_1d(w_fc_int[0])}
}};

// Fully Connected Layer Bias
const int32_t FC_BIAS = {b_fc_int[0]};

#endif // CHESS_NET_HPP
"""

    with open(output_path, "w") as f:
        f.write(hpp_content)
        
    print(f"Successfully generated header file: {output_path}")
    print(f"Weights scaled by {scale}, Output Bias scaled by {scale * scale}")

if __name__ == "__main__":
    export_to_hpp(model_path="sillynet3.pt", output_path="sillynet3.hpp", scale=64)