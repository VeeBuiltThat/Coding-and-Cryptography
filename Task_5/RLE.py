def rle_compress(text):
    if not text:
        return ""

    result = []
    count = 1

# counts and shows how many times a character repeats in a row
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            result.append(text[i - 1] + str(count))
            count = 1

    result.append(text[-1] + str(count))
    return "".join(result)

# just repeat the character as many times as needed
def rle_decompress(encoded):
    result = ""
    i = 0

    while i < len(encoded):
        char = encoded[i]
        i += 1
        num = ""

        while i < len(encoded) and encoded[i].isdigit():
            num += encoded[i]
            i += 1

        result += char * int(num)

    return result
pass
 
if __name__ == "__main__":
    s = "AAABBBCCCCDD"
    comp = rle_compress(s)
    decomp = rle_decompress(comp)
    print("Original:", s)
    print("Compressed:", comp)
    print("Decompressed:", decomp)
    # it will keep goin until the string is expanded fully