from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


def png_optimization_analysis(input_file='my_logo.png'):
    try:
        img = Image.open(input_file)
    except FileNotFoundError:
        print("Please create 'my_logo.png' first.")
        return

    original_size = os.path.getsize(input_file)
    print(f"Original mode: {img.mode}, metadata: {img.info}, size: {original_size} bytes")

    img.save('stripped.png')
    stripped_size = os.path.getsize('stripped.png')


    indexed_img = img.quantize()
    indexed_img.save('indexed.png')
    indexed_size = os.path.getsize('indexed.png')

    print("--- Results ---")
    print(f"Original: {original_size} bytes")
    print(f"Stripped: {stripped_size} bytes, Savings: {original_size - stripped_size}")
    print(f"Indexed: {indexed_size} bytes, Savings: {original_size - indexed_size}")


def png_filters_visualization(input_file='my_photo.jpg'):
    try:
        img = Image.open(input_file).convert('L')
    except FileNotFoundError:
        print("Please add 'my_photo.jpg' file first.")
        return

    original_data = np.array(img, dtype=np.int16)
    # Sub Filter (X-A)
    sub_filtered = original_data.copy()
    for r in range(original_data.shape[0]):
        for c in range(1, original_data.shape[1]):
            sub_filtered[r, c] = original_data[r, c] - original_data[r, c-1]


    up_filtered = original_data.copy()
    for r in range(1, original_data.shape[0]):
        for c in range(original_data.shape[1]):
            up_filtered[r, c] = original_data[r, c] - original_data[r-1, c]


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(original_data, cmap='gray'); ax1.set_title('Original')
    ax2.imshow(sub_filtered + 128, cmap='gray', vmin=0, vmax=255); ax2.set_title('Sub Filter')
    ax3.imshow(up_filtered + 128, cmap='gray', vmin=0, vmax=255); ax3.set_title('Up Filter')
    plt.show()

png_optimization_analysis()
png_filters_visualization()