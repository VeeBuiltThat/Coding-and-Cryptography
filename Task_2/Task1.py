from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


message = input("Enter a message to encrypt: ").encode()

key = get_random_bytes(8)
cipher = DES.new(key, DES.MODE_ECB)

padded_msg = pad(message, DES.block_size)

encrypted = cipher.encrypt(padded_msg)
print("Encrypted:", encrypted)

decipher = DES.new(key, DES.MODE_ECB)
decrypted = unpad(decipher.decrypt(encrypted), DES.block_size)
print("Decrypted:", decrypted.decode())
