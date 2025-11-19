from PIL import Image

DELIMITER = chr(0)   

def stega_decrypt_delim():
    img = Image.open(input("path to image: ").strip()).convert("RGB")
    pix = img.load()
    fpath = input('path to keys: ').strip()
    with open(fpath,'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    message_chars = []
    for line in lines:
        x_str, y_str = line.split(',')
        x, y = int(x_str), int(y_str)
        val = pix[(x, y)][0]   
        if val == 0:           
            break
        message_chars.append(chr(val))

    return ''.join(message_chars)

if __name__ == "__main__":
    print("your message:", stega_decrypt_delim())
