def lzw_compress(text):
    dictionary = {chr(i): i for i in range(256)} # ASCII characters
    next_code = 256 # each has that number
    current = ""
    result = []

    for char in text:
        combined = current + char
        if combined in dictionary:
            current = combined
        else:
            result.append(dictionary[current])
            dictionary[combined] = next_code
            next_code += 1
            current = char

    if current:
        result.append(dictionary[current])

    return result

def lzw_decompress(codes):
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256

    prev = dictionary[codes[0]]
    result = prev

    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = prev + prev[0]

        result += entry
        dictionary[next_code] = prev + entry[0]
        next_code += 1
        prev = entry

    return result

if __name__ == "__main__":
    s = "TOBEORNOTTOBEORTOBEORNOT"
    comp = lzw_compress(s)
    decomp = lzw_decompress(comp)
    print("Original:", s)
    print("Compressed:", comp)
    print("Decompressed:", decomp)