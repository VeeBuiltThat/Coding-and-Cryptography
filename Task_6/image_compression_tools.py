import os
import sys
import heapq
import pickle
from collections import defaultdict
from PIL import Image
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from skimage.metrics import mean_squared_error
except Exception:
    mean_squared_error = None


class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value  
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree_from_freq(freq_map):
    pq = []
    for val, f in freq_map.items():
        heapq.heappush(pq, HuffmanNode(val, f))
    if not pq:
        return None
    while len(pq) > 1:
        a = heapq.heappop(pq)
        b = heapq.heappop(pq)
        parent = HuffmanNode(None, a.freq + b.freq)
        parent.left = a
        parent.right = b
        heapq.heappush(pq, parent)
    return pq[0]

def generate_codes_from_tree(root):
    codes = {}
    if root is None:
        return codes
    def dfs(node, prefix):
        if node.value is not None:
            codes[node.value] = prefix or "0"  
            return
        dfs(node.left, prefix + "0")
        dfs(node.right, prefix + "1")
    dfs(root, "")
    return codes

def compress_hybrid(input_file, output_file, quant_factor=10):
    """
    1) Convert to grayscale
    2) Quantize each pixel by //quant_factor (lossy)
    3) Flatten and apply RLE producing list of (value, count)
    4) Split runs >255 into chunks
    5) Build Huffman on the RLE symbols (tuples)
    6) Encode the sequence and write file containing header + bitstream
    Header stored via pickle: {'shape': shape, 'codes': codes} where codes keys are tuples
    """
    print(f"Hybrid compress: {input_file} -> {output_file} (quant={quant_factor})")
    image = Image.open(input_file).convert('L')
    arr = np.array(image)
    shape = arr.shape

    q = (arr // quant_factor).astype(np.uint8)

    flat = q.flatten()
    rle = []
    if flat.size == 0:
        print("Empty image.")
        return
    cur_val = int(flat[0])
    cnt = 1
    for pix in flat[1:]:
        pix = int(pix)
        if pix == cur_val and cnt < 65535: 
            cnt += 1
        else:
            rle.append((cur_val, cnt))
            cur_val = pix
            cnt = 1
    rle.append((cur_val, cnt))

    freq = defaultdict(int)
    for sym in rle:
        freq[sym] += 1

    root = build_huffman_tree_from_freq(freq)
    codes = generate_codes_from_tree(root)

    bitstr = "".join(codes[sym] for sym in rle)

    with open(output_file, 'wb') as f:
        header = {'shape': shape, 'quant_factor': quant_factor, 'codes': codes}
        pickle.dump(header, f)

        padding = (8 - (len(bitstr) % 8)) % 8
        f.write(bytes([padding]))
        if padding:
            bitstr += '0' * padding

        ba = bytearray()
        for i in range(0, len(bitstr), 8):
            ba.append(int(bitstr[i:i+8], 2))
        f.write(ba)

    print("Hybrid compression written. Original size:", os.path.getsize(input_file),
          "Compressed size:", os.path.getsize(output_file))

def decompress_hybrid(input_file, output_file):
    """
    Read header and bitstream, decode to list of (value,count) tuples,
    expand RLE to quantized flattened pixels, unquantize by *quant_factor, reshape and save.
    """
    print(f"Hybrid decompress: {input_file} -> {output_file}")
    with open(input_file, 'rb') as f:
        header = pickle.load(f)
        shape = tuple(header['shape'])
        quant_factor = header['quant_factor']
        codes = header['codes']

        inverted = {v:k for k,v in codes.items()}
        padding = int.from_bytes(f.read(1), 'big')
        data = f.read()
        bitstr = ''.join(f'{byte:08b}' for byte in data)
        if padding:
            bitstr = bitstr[:-padding]

        cur = ""
        rle = []
        for b in bitstr:
            cur += b
            if cur in inverted:
                rle.append(inverted[cur])
                cur = ""


    flat = []
    for val, cnt in rle:
        flat.extend([int(val)] * int(cnt))

    if len(flat) != shape[0] * shape[1]:
        print(f"Warning: decompressed pixel count {len(flat)} != expected {shape[0]*shape[1]}.")

        if len(flat) > shape[0]*shape[1]:
            flat = flat[:shape[0]*shape[1]]
        else:
            flat.extend([0] * (shape[0]*shape[1] - len(flat)))

    arr = (np.array(flat, dtype=np.uint8).reshape(shape) * quant_factor).astype(np.uint8)
    Image.fromarray(arr).save(output_file)
    print("Hybrid decompressed image saved.")

def compress_rle_v2(input_file, output_file):
    """
    Binary format:
    - 2 bytes: height (big-endian)
    - 2 bytes: width  (big-endian)
    - then pairs: [value (1 byte), count (1 byte)] repeated until EOF
    If count > 255, we split into multiple pairs.
    """
    print(f"RLEv2 compress: {input_file} -> {output_file}")
    image = Image.open(input_file).convert('L')
    arr = np.array(image)
    h, w = arr.shape
    flat = arr.flatten()

    with open(output_file, 'wb') as f:
        f.write(h.to_bytes(2, 'big'))
        f.write(w.to_bytes(2, 'big'))
        if flat.size == 0:
            return
        cur = int(flat[0])
        cnt = 1
        for pix in flat[1:]:
            pix = int(pix)
            if pix == cur:
                cnt += 1
                if cnt == 256:
                    f.write(bytes([cur, 255]))
                    cnt = 1
            else:

                while cnt > 255:
                    f.write(bytes([cur, 255]))
                    cnt -= 255
                f.write(bytes([cur, cnt]))
                cur = pix
                cnt = 1

        while cnt > 255:
            f.write(bytes([cur, 255]))
            cnt -= 255
        f.write(bytes([cur, cnt]))
    print("RLE v2 written. sizes:", os.path.getsize(input_file), "->", os.path.getsize(output_file))

def decompress_rle_v2(input_file, output_file):
    print(f"RLEv2 decompress: {input_file} -> {output_file}")
    with open(input_file, 'rb') as f:
        header = f.read(4)
        if len(header) < 4:
            print("Invalid file.")
            return
        h = int.from_bytes(header[0:2], 'big')
        w = int.from_bytes(header[2:4], 'big')
        flat = []
        while True:
            pair = f.read(2)
            if not pair or len(pair) < 2:
                break
            value = pair[0]
            cnt = pair[1]
            flat.extend([value] * cnt)
    if len(flat) != h*w:
        print(f"Warning: decompressed length {len(flat)} != expected {h*w}. Adjusting.")
        if len(flat) > h*w:
            flat = flat[:h*w]
        else:
            flat.extend([0]*(h*w - len(flat)))
    arr = np.array(flat, dtype=np.uint8).reshape((h,w))
    Image.fromarray(arr).save(output_file)
    print("RLEv2 decompressed image saved.")


def create_test_simple(path="test_simple.png", w=200, h=200):
    arr = np.zeros((h,w), dtype=np.uint8)
    arr[0:100, :] = 50
    arr[:, 150:160] = 255
    Image.fromarray(arr).save(path)
    print("Test image saved to", path)

def compress_rle_scan(input_file, output_file, mode='horizontal'):
    """
    Stores:
     - 2 bytes height, 2 bytes width, 1 byte mode (1 horiz, 2 vert)
     - For each scan line (row or column):
         - 2 bytes: number of pairs for that line (N)
         - then N pairs of (value, count) single bytes each
    """
    print(f"RLE scan compress: {input_file} -> {output_file} mode={mode}")
    image = Image.open(input_file).convert('L')
    arr = np.array(image)
    h,w = arr.shape

    if mode == 'horizontal':
        lines = [arr[row,:] for row in range(h)]
        mode_byte = 1
    elif mode == 'vertical':
        lines = [arr[:,col] for col in range(w)]
        mode_byte = 2
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'")

    with open(output_file, 'wb') as f:
        f.write(h.to_bytes(2,'big'))
        f.write(w.to_bytes(2,'big'))
        f.write(bytes([mode_byte]))
        for line in lines:

            pairs = []
            cur = int(line[0])
            cnt = 1
            for pix in line[1:]:
                pix = int(pix)
                if pix == cur:
                    cnt += 1
                    if cnt == 256:
                        pairs.append((cur,255)); cnt = 1
                else:
                    while cnt > 255:
                        pairs.append((cur,255)); cnt -= 255
                    pairs.append((cur,cnt))
                    cur = pix; cnt = 1
            while cnt > 255:
                pairs.append((cur,255)); cnt -= 255
            pairs.append((cur,cnt))

            f.write(len(pairs).to_bytes(2,'big'))

            for val,c in pairs:
                f.write(bytes([val, c]))

    print("RLE-scan file written:", os.path.getsize(output_file), "bytes")

def decompress_rle_scan(input_file, output_file):
    with open(input_file,'rb') as f:
        header = f.read(5)
        if len(header) < 5:
            print("Invalid file")
            return
        h = int.from_bytes(header[0:2],'big')
        w = int.from_bytes(header[2:4],'big')
        mode_byte = header[4]
        if mode_byte == 1:
            horizontal = True
            lines_count = h
        else:
            horizontal = False
            lines_count = w

        if horizontal:
            result = np.zeros((h,w), dtype=np.uint8)
        else:
            result = np.zeros((h,w), dtype=np.uint8)

        for line_idx in range(lines_count):
            n_pairs = int.from_bytes(f.read(2), 'big')
            flat_line = []
            for _ in range(n_pairs):
                pair = f.read(2)
                val = pair[0]; cnt = pair[1]
                flat_line.extend([val]*cnt)

            if horizontal:
                row = np.array(flat_line[:w], dtype=np.uint8)
                if row.size < w:
                    row = np.concatenate([row, np.zeros(w-row.size,dtype=np.uint8)])
                result[line_idx,:] = row
            else:
                col = np.array(flat_line[:h], dtype=np.uint8)
                if col.size < h:
                    col = np.concatenate([col, np.zeros(h-col.size,dtype=np.uint8)])
                result[:,line_idx] = col

    Image.fromarray(result).save(output_file)
    print("RLE-scan decompressed saved to", output_file)

def compare_rle_scan(test_image="test_simple.png", natural_image="sample.png"):

    if not os.path.exists(test_image):
        create_test_simple(test_image)

    hfile = "test_simple.hrle"
    vfile = "test_simple.vrle"
    compress_rle_scan(test_image, hfile, 'horizontal')
    compress_rle_scan(test_image, vfile, 'vertical')
    size_h = os.path.getsize(hfile)
    size_v = os.path.getsize(vfile)
    print("Sizes for", test_image, "-> horizontal:", size_h, "vertical:", size_v)
    
    decompress_rle_scan(hfile, "test_simple_hrle_decomp.png")
    decompress_rle_scan(vfile, "test_simple_vrle_decomp.png")

    if os.path.exists(natural_image):
        hfile2 = "sample.hrle"
        vfile2 = "sample.vrle"
        compress_rle_scan(natural_image, hfile2, 'horizontal')
        compress_rle_scan(natural_image, vfile2, 'vertical')
        print("Sizes for natural image -> horizontal:", os.path.getsize(hfile2),
              "vertical:", os.path.getsize(vfile2))
    else:
        print("Natural image not found; skipped natural photo test.")


def jpeg_quality_analysis(input_file, qualities=None, plot=False):
    if mean_squared_error is None:
        print("skimage.metrics.mean_squared_error not available. Install scikit-image.")
        return
    if qualities is None:
        qualities = [5,10,15,25,50,75,85,95]
    orig_img = Image.open(input_file).convert('RGB')
    orig_arr = np.array(orig_img)
    file_sizes = []
    errors = []
    temp_files = []
    for q in qualities:
        out = f"temp_q{q}.jpg"
        orig_img.convert('RGB').save(out, "JPEG", quality=q)
        temp_files.append(out)
        size = os.path.getsize(out)
        file_sizes.append(size)
        recon = np.array(Image.open(out).convert('RGB'))
        mse = mean_squared_error(orig_arr.astype(np.float32), recon.astype(np.float32))
        errors.append(mse)
        print(f"Quality {q}: size={size} bytes, MSE={mse:.2f}")

    print("\nQuality | Size (bytes) | MSE")
    for q,s,e in zip(qualities, file_sizes, errors):
        print(f"{q:3}     {s:10}       {e:.2f}")

    if plot and plt is not None:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(qualities, file_sizes, marker='o')
        plt.title("File Size vs Quality")
        plt.xlabel("Quality"); plt.ylabel("Size (bytes)")
        plt.subplot(1,3,2)
        plt.plot(qualities, errors, marker='o')
        plt.title("MSE vs Quality")
        plt.xlabel("Quality"); plt.ylabel("MSE")
        plt.subplot(1,3,3)
        plt.plot(file_sizes, errors, marker='o')
        plt.title("Rate-Distortion (MSE vs Size)")
        plt.xlabel("Size (bytes)"); plt.ylabel("MSE")
        plt.tight_layout()
        plt.show()


def print_help():
    print(__doc__)

def main(argv):
    if len(argv) < 2:
        print_help(); return
    cmd = argv[1].lower()
    if cmd == 'hybrid':
        if len(argv) < 4:
            print("Usage: hybrid [compress|decompress] in out"); return
        action = argv[2]; infile = argv[3]; outfile = argv[4]
        if action == 'compress':
            compress_hybrid(infile, outfile, quant_factor=10)
        else:
            decompress_hybrid(infile, outfile)
    elif cmd == 'rle_v2':
        if len(argv) < 4:
            print("Usage: rle_v2 [compress|decompress] in out"); return
        action = argv[2]; infile = argv[3]; outfile = argv[4]
        if action == 'compress':
            compress_rle_v2(infile, outfile)
        else:
            decompress_rle_v2(infile, outfile)
    elif cmd == 'rle_scan':
        if len(argv) < 3:
            print("Usage: rle_scan [compress|decompress|compare] ..."); return
        action = argv[2]
        if action == 'compare':
            test_img = argv[3] if len(argv)>3 else "test_simple.png"
            compare_rle_scan(test_img)
        elif action == 'compress':
            mode = argv[4] if len(argv)>4 else 'horizontal'
            compress_rle_scan(argv[3], argv[4], mode)
        elif action == 'decompress':
            decompress_rle_scan(argv[3], argv[4])
    elif cmd == 'jpeg_analysis':
        if len(argv) < 3:
            print("Usage: jpeg_analysis input.png [plot]"); return
        plot = (len(argv)>3 and argv[3].lower()=='plot')
        jpeg_quality_analysis(argv[2], plot=plot)
    else:
        print_help()

if __name__ == "__main__":
    main(sys.argv)
