# 18: DES Encryption and Decryption
# Description: Encrypt and decrypt a message using the DES algorithm in Python. Explain the chosen encryption mode.

from Crypto.Cipher import DES
from Crypto.Random import get_random_bytes # too bothered to figure out why this is grey
import base64

def pad(text):
    while len(text) % 8 != 0:
        text += " "
    return text

def des_encrypt(key, plaintext):
    plaintext = pad(plaintext)
    cipher = DES.new(key, DES.MODE_CBC)
    iv = cipher.iv
    encrypted_bytes = cipher.encrypt(plaintext.encode('utf-8'))
    return base64.b64encode(iv + encrypted_bytes).decode('utf-8')

def des_decrypt(key, ciphertext_b64):
    ciphertext = base64.b64decode(ciphertext_b64)
    iv = ciphertext[:8]               
    encrypted_message = ciphertext[8:]
    
    cipher = DES.new(key, DES.MODE_CBC, iv=iv)
    decrypted = cipher.decrypt(encrypted_message).decode('utf-8')
    return decrypted.rstrip()       

key = b"8bytekey"     
message = "HELLO DES ENCRYPTION"

encrypted = des_encrypt(key, message)
decrypted = des_decrypt(key, encrypted)

print("Original :", message)
print("Encrypted:", encrypted)
print("Decrypted:", decrypted)
# testing, dont mind this