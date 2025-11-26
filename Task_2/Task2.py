## Encyrpting several example messages

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
    padded = pad(msg, DES.block_size) # gotta be padded because DES works with 8-byte blocks, should be adjusted to 8-16 bytes
    encrypted = cipher.encrypt(padded) # same DES key
    
    print(f"\nOriginal message: {msg}")
    print(f"Padded message ({len(padded)} bytes): {padded}")
    print(f"Encrypted ({len(encrypted)} bytes): {encrypted}")
# if this works, its a miriacle...