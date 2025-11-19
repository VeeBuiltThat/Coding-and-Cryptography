import numpy as np
from scipy.fftpack import dct

block = np.array([
    [150, 140, 130, 120, 120, 120, 120, 120],
    [140, 130, 120, 110, 110, 110, 110, 110],
    [130, 120, 110, 100, 100, 100, 100, 100],
    [120, 110, 100, 90, 90, 90, 90, 90],
    [120, 110, 100, 90, 90, 90, 90, 90],
    [120, 110, 100, 90, 90, 90, 90, 90],
    [120, 110, 100, 90, 90, 90, 90, 90],
    [120, 110, 100, 90, 90, 90, 90, 90]
], dtype=float)

block_shifted = block - 128
dct_block = dct(dct(block_shifted.T, norm='ortho').T, norm='ortho')
print("--- 8x8 DCT Coefficients ---")
print(np.round(dct_block, 2))

def zigzag_scan(matrix):
    rows, cols = matrix.shape
    result = np.zeros(rows*cols)
    r = c = index = 0
    going_down = True
    while index < rows*cols:
        result[index] = matrix[r, c]
        index += 1
        if going_down:
            if c == 0 or r == rows-1:
                going_down = False
                if r == rows-1: c += 1
                else: r += 1
            else: r += 1; c -= 1
        else:
            if r == 0 or c == cols-1:
                going_down = True
                if c == cols-1: r += 1
                else: c += 1
            else: r -= 1; c += 1
    return result

zz_array = zigzag_scan(dct_block)
print("\n--- Zig-Zag 1D Array ---")
print(np.round(zz_array, 2))
