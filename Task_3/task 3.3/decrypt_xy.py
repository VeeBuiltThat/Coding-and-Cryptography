from PIL import Image

def stega_decrypt_xy():
    a = []
    keys = []
    img = Image.open(input("path to image: ").strip()).convert("RGB")
    pix = img.load()
    fpath = input('path to keys: ').strip()
    with open(fpath,'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        # parse "x,y"
        parts = line.split(',')
        if len(parts) == 2:
            x = int(parts[0])
            y = int(parts[1])
            keys.append((x, y))

    for key in keys:
        a.append(pix[tuple(key)][0]) 

    return ''.join([chr(elem) for elem in a])

if __name__ == "__main__":
    print("your message:", stega_decrypt_xy())
