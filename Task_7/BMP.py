import struct

def create_bmp(filename='python_gradient.bmp', width=256, height=256, bits_per_pixel=8):

    padding_size = (4 - (width * bits_per_pixel // 8) % 4) % 4
    image_size = (width * bits_per_pixel // 8 + padding_size) * height
    palette_size = 256 * 4
    file_offset_bits = 14 + 40 + palette_size
    file_size = file_offset_bits + image_size

    file_header = b'BM'
    file_header += struct.pack('<L', file_size)
    file_header += struct.pack('<HH', 0, 0)
    file_header += struct.pack('<L', file_offset_bits)

    info_header = struct.pack('<L', 40)
    info_header += struct.pack('<LL', width, height)
    info_header += struct.pack('<H', 1)
    info_header += struct.pack('<H', bits_per_pixel)
    info_header += struct.pack('<L', 0)
    info_header += struct.pack('<L', image_size)
    info_header += struct.pack('<LL', 0, 0)
    info_header += struct.pack('<L', 256)
    info_header += struct.pack('<L', 0)

    palette = bytearray()
    for i in range(256):
        palette += struct.pack('<BBBB', i, i, i, 0)

    pixels = bytearray()
    padding = b'\x00' * padding_size
    for i in range(height):
        for j in range(width):
            pixels.append(j)  
        pixels += padding

    with open(filename, 'wb') as f:
        f.write(file_header)
        f.write(info_header)
        f.write(palette)
        f.write(pixels)

    print(f"Created '{filename}' ({file_size} bytes)")

create_bmp()
