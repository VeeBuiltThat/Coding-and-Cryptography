# 17: Implement Run-Length Encoding (RLE)
# Description: Write a Python function to compress a string using Run-Length Encoding and decompress it back to the original string.
def rle_compress(data: str) -> str:
    if not data:
        return ""

    compressed = []
    count = 1

    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            compressed.append(str(count) + data[i - 1])
            count = 1

    compressed.append(str(count) + data[-1])
    return "".join(compressed)


def rle_decompress(encoded: str) -> str:
    if not encoded:
        return ""

    decompressed = []
    count = ""

    for char in encoded:
        if char.isdigit():
            count += char
        else:
            decompressed.append(char * int(count))
            count = ""

    return "".join(decompressed)


#this is a tester, dont mind this
original = "AAABBBCCCCDD"
compressed = rle_compress(original)
decompressed = rle_decompress(compressed)

print("Original:", original)
print("Compressed:", compressed)
print("Decompressed:", decompressed)
