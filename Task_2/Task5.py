from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


key = get_random_bytes(8)
cipher = DES.new(key, DES.MODE_ECB)

with open("example.txt", "rb") as f:
    data = f.read()

padded = pad(data, DES.block_size)
encrypted = cipher.encrypt(padded)

with open("example.txt.enc", "wb") as f:
    f.write(encrypted)

print("File encrypted → example.txt.enc")

decipher = DES.new(key, DES.MODE_ECB)

with open("example.txt.enc", "rb") as f:
    enc_data = f.read()

decrypted = unpad(decipher.decrypt(enc_data), DES.block_size)

with open("example_decrypted.txt", "wb") as f:
    f.write(decrypted)

print("File decrypted → example_decrypted.txt")
