def rle_compress(text):
    if not text:
        return ""

    result = []
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            result.append(text[i - 1] + str(count))
            count = 1

    result.append(text[-1] + str(count))
    return "".join(result)


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
