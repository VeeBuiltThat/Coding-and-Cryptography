# Coding-and-Cryptography
Coding and Cryptography tasks

# Image Compression and File Format Analysis
Includes hands-on exercises for BMP, PNG, and JPEG formats, covering both lossless and lossy methods. Students explore headers, pixel data, palettes, filters, chroma subsampling, DCT, and zig-zag scanning.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Requirements](#requirements)  
3. [Task Set 1: BMP File Structure](#task-set-1-bmp-file-structure)  
4. [Task Set 2: PNG Lossless Compression & Structure](#task-set-2-png-lossless-compression--structure)  
5. [Task Set 3: JPEG Lossy Compression & Artifacts](#task-set-3-jpeg-lossy-compression--artifacts)  
6. [Task Set 4: JPEG Core – DCT and Zig-Zag Scanning](#task-set-4-jpeg-core--dct-and-zig-zag-scanning)  
7. [How to Run](#how-to-run)  
8. [References](#references)  

## Project Overview
The goal of this project is to understand **how different image formats store pixel data, how compression works, and how visual artifacts arise** from lossy compression.

Key concepts explored:

- File headers and pixel storage (BMP)
- Palette/indexed color and PNG filtering (PNG)
- Luma/chroma separation and chroma subsampling (JPEG)
- Discrete Cosine Transform (DCT) and Zig-Zag scanning for JPEG

## Requirements
Install the necessary Python libraries:

```bash
pip install pillow numpy matplotlib scipy
