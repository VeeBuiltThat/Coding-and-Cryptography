from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

key = get_random_bytes(8)
cipher = DES.new(key, DES.MODE_ECB)

messages = [
    b"A",                 
    b"Hello",             
    b"DES example text!"   
]

for msg in messages:
    padded = pad(msg, DES.block_size)
    encrypted = cipher.encrypt(padded)
    
    print(f"\nOriginal message: {msg}")
    print(f"Padded message ({len(padded)} bytes): {padded}")
    print(f"Encrypted ({len(encrypted)} bytes): {encrypted}")
