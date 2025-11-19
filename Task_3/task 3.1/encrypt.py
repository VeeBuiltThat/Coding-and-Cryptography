from PIL import Image, ImageDraw
from random import randint

def stega_encrypt():
    keys = []
    img_path = input("path to image: ").strip()
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    pix = img.load()

    f = open('keys.txt','w')

    text = input("text here: ")

    for elem in ([ord(ch) for ch in text]):
        key = (randint(0, width-1), randint(0, height-1))
        r, g, b = pix[key]
        draw.point(key, (elem, g, b))
        f.write(str(key) + '\n')

    print('keys were written to the keys.txt file')
    img.save("newimage.png", "PNG")
    f.close()

if __name__ == "__main__":
    stega_encrypt()
