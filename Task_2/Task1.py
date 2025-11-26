# Encrypting and Decrypting a message (ECB mode)

from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

message = input("Enter a message to encrypt: ").encode() # <-- Enter message here

key = get_random_bytes(8)
cipher = DES.new(key, DES.MODE_ECB) # above and this line, generates a random 8-byte DES

padded_msg = pad(message, DES.block_size) # padding to match the length, apperently its a requirement for DES? (ask prof to elaborate pls)

encrypted = cipher.encrypt(padded_msg) 
print("Encrypted:", encrypted)

decipher = DES.new(key, DES.MODE_ECB) # using the same key
decrypted = unpad(decipher.decrypt(encrypted), DES.block_size) # unpad = removes padding
print("Decrypted:", decrypted.decode())

# Need to ask prof to elaborate on DES requirements, line 10