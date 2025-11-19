from PIL import Image
import matplotlib.pyplot as plt


def ycbcr_chroma_subsampling(input_file='my_photo.jpg'):
    try:
        img = Image.open(input_file)
    except FileNotFoundError:
        print("Please add 'my_photo.jpg' file first.")
        return

    img_ycbcr = img.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.imshow(img); ax1.set_title('Original RGB')
    ax2.imshow(y, cmap='gray'); ax2.set_title('Y (Luma)')
    ax3.imshow(cb, cmap='gray'); ax3.set_title('Cb (Chroma Blue)')
    ax4.imshow(cr, cmap='gray'); ax4.set_title('Cr (Chroma Red)')
    plt.show()

    size = y.size
    cb_small = cb.resize((size[0]//2, size[1]//2)).resize(size, Image.Resampling.BILINEAR)
    cr_small = cr.resize((size[0]//2, size[1]//2)).resize(size, Image.Resampling.BILINEAR)
    img_resampled = Image.merge('YCbCr', (y, cb_small, cr_small)).convert('RGB')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img); ax1.set_title('Original')
    ax2.imshow(img_resampled); ax2.set_title('Chroma Subsampled 4:2:0')
    plt.show()

# Uncomment to run
ycbcr_chroma_subsampling()
