#ifndef CHESS_NET_HPP
#define CHESS_NET_HPP

#include <cstdint>

// Quantization Parameters
const int SCALE_FACTOR = 64;
const int SCALE_FACTOR_SQ = 4096;

// Convolutional Weights: Shape [12][3][3] (in_channels x kernel_y x kernel_x)
// Output channels = 1, so the outermost dimension is omitted.
const int16_t CONV_WEIGHTS[12][3][3] = {
    {
        {40, 1, 30},
        {12, -8, 6},
        {54, 25, 36}
    },
    {
        {52, 32, 32},
        {31, -32, 27},
        {96, 32, 62}
    },
    {
        {62, 28, 44},
        {37, -34, 26},
        {110, 41, 72}
    },
    {
        {109, 37, 77},
        {55, -54, 39},
        {182, 71, 120}
    },
    {
        {171, 64, 123},
        {85, -86, 64},
        {294, 109, 189}
    },
    {
        {-30, -13, -27},
        {-16, 6, -3},
        {-60, -16, -48}
    },
    {
        {-30, -10, -22},
        {-8, 4, -5},
        {-63, -10, -41}
    },
    {
        {-44, -31, -30},
        {-44, 59, -39},
        {-76, -59, -42}
    },
    {
        {-55, -26, -43},
        {-38, 41, -26},
        {-100, -47, -62}
    },
    {
        {-82, -69, -57},
        {-78, 90, -58},
        {-168, -78, -107}
    },
    {
        {-140, -80, -102},
        {-104, 111, -66},
        {-258, -112, -174}
    },
    {
        {18, 10, 21},
        {16, -6, 3},
        {38, 14, 31}
    }
};

// Convolutional Bias
const int32_t CONV_BIAS = 365;

// Fully Connected Layer Weights: Shape [64] (Flattened 8x8 input)
const int16_t FC_WEIGHTS[64] = {
    -44, 15, 36, -12, -24, 13, 24, -5, -48, 51, 84, 14, 0, 53, 54, 16, -66, 45, 82, 4, -5, 48, 43, -1, -106, -4, 31, -47, -61, -7, -4, -35, -92, 16, 56, -29, -47, 17, 18, -12, -34, 99, 140, 52, 41, 103, 100, 38, -53, 49, 75, 11, 3, 48, 45, 3, -117, -92, -75, -116, -120, -91, -88, -73
};

// Fully Connected Layer Bias
const int32_t FC_BIAS = -6616;

#endif // CHESS_NET_HPP
