from PIL import Image, ImageDraw
from random import randint

DELIMITER = chr(0)  

def stega_encrypt_delim():
    img_path = input("path to image: ").strip()
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    pix = img.load()
    f = open('keys_delim.txt','w')

    text = input("text here: ")
    text_with_delim = text + DELIMITER

    for ch in text_with_delim:
        elem = ord(ch)
        key = (randint(0, width-1), randint(0, height-1))
        r, g, b = pix[key]
        draw.point(key, (elem, g, b))
        f.write(f"{key[0]},{key[1]}\n")  

    print('keys were written to keys_delim.txt')
    img.save("newimage_delim.png", "PNG")
    f.close()

if __name__ == "__main__":
    stega_encrypt_delim()
