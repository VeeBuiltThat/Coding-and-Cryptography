## using DES in CBC mode

from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


message = b"DES CBC Mode Example!"

key = get_random_bytes(8)
iv = get_random_bytes(8)     

cipher = DES.new(key, DES.MODE_CBC, iv)
encrypted = cipher.encrypt(pad(message, DES.block_size))

print("Encrypted (CBC):", encrypted)

# Decrypt
decipher = DES.new(key, DES.MODE_CBC, iv) # CBC method, pads the message and encrypts
decrypted = unpad(decipher.decrypt(encrypted), DES.block_size) # Same key, same IV

print("Decrypted:", decrypted)
