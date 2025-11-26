import struct

def create_bmp(filename='python_gradient.bmp', width=256, height=256, bits_per_pixel=24):
    padding_size = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + padding_size) * height
    file_offset_bits = 14 + 40
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
    info_header += struct.pack('<L', 0)
    info_header += struct.pack('<L', 0)
    # line 21 and 22 are colors   

    pixels = bytearray()
    padding = b'\x00' * padding_size
    for row in range(height):
        for col in range(width):
            r = int(col * 127 / (width - 1)) # col color
            g = int(row * 190 / (height - 1)) # row color
            b = 128 # constant color
            pixels += struct.pack('<BBB', b, g, r)
        pixels += padding

    with open(filename, 'wb') as f:
        f.write(file_header)
        f.write(info_header)
        f.write(pixels)

    print(f"Created '{filename}' ({file_size} bytes)")

create_bmp()
