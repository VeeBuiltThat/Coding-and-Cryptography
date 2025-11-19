from PIL import Image, ImageDraw
from random import randint

def stega_encrypt_xy():
    img_path = input("path to image: ").strip()
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    pix = img.load()
    f = open('keys_xy.txt','w')

    text = input("text here: ")

    for elem in ([ord(ch) for ch in text]):
        key = (randint(0, width-1), randint(0, height-1))
        r, g, b = pix[key]
        draw.point(key, (elem, g, b))
        f.write(f"{key[0]},{key[1]}\n")

    print('keys were written to keys_xy.txt')
    img.save("newimage_xy.png", "PNG")
    f.close()

if __name__ == "__main__":
    stega_encrypt_xy()
