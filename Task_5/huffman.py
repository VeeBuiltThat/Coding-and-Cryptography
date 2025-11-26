import heapq

# frequencies
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encode(text):
    if not text:
        return "", {}

    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

# priority queue thing, or whatever it was called
    heap = [Node(char, freq[char]) for char in freq]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    root = heap[0]

    codes = {}
    def generate_codes(node, current_code):
        if node.char is not None:
            codes[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
        # basically, it got the flavor in each binary code

    generate_codes(root, "")


    encoded = "".join(codes[ch] for ch in text)
    return encoded, codes, root

def huffman_decode(encoded, root):
    decoded = ""
    node = root

    for bit in encoded:
        if bit == "0":
            node = node.left
        else:
            node = node.right

        if node.char is not None:
            decoded += node.char
            node = root

    return decoded

if __name__ == "__main__":
    s = "this is an example for huffman encoding"
    encoded, codes, root = huffman_encode(s)
    decoded = huffman_decode(encoded, root)
    print("Original:", s)
    print("Encoded:", encoded)
    print("Codes:", codes)
    print("Decoded:", decoded)
