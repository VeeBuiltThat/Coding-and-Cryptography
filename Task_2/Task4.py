## Brute-forcing a weak DES key

from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

plaintext = b"TEST1234" # turns out this is called a plaintext block
real_key = b"\x00\x00\x00\x00\x00\x00\x00\x5A"  # as long as the last byte is not a zero....you good

cipher = DES.new(real_key, DES.MODE_ECB)
encrypted = cipher.encrypt(plaintext)

print("Target Ciphertext:", encrypted)
# basically, plaintext with real key, and you good to go with encryption 

for k in range(256): # the number is the amount of possibilities, btw, this is important, dont forget woman
    test_key = b"\x00\x00\x00\x00\x00\x00\x00" + bytes([k])
    try:
        c = DES.new(test_key, DES.MODE_ECB) # try to remember to continue tomorrow 
        decrypted = c.decrypt(encrypted)
        
        if decrypted == plaintext:
            print("\nKey Found:", test_key)
            print("Decrypted:", decrypted)
            break
    except:
        continue
