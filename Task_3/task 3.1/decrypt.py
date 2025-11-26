# Redchannel keys as printed tuple strings

from PIL import Image
from re import findall

def stega_decrypt(): # adds character code in redchannel of random chosen pixels 
    a = []
    keys = []
    img = Image.open(input("path to image: ").strip()).convert("RGB")
    pix = img.load()
    fpath = input('path to keys: ').strip()
    with open(fpath,'r') as f:
        y = str([line.strip() for line in f])

    xs = findall(r'\((\d+)\,', y)
    ys = findall(r'\,\s?(\d+)\)', y)

    for i in range(len(xs)):
        keys.append((int(xs[i]), int(ys[i])))

    for key in keys:
        a.append(pix[tuple(key)][0]) 

    return ''.join([chr(elem) for elem in a])

if __name__ == "__main__":
    print("your message:", stega_decrypt())
