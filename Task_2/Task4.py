from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad



plaintext = b"TEST1234"   
real_key = b"\x00\x00\x00\x00\x00\x00\x00\x5A"  

cipher = DES.new(real_key, DES.MODE_ECB)
encrypted = cipher.encrypt(plaintext)

print("Target Ciphertext:", encrypted)

for k in range(256):
    test_key = b"\x00\x00\x00\x00\x00\x00\x00" + bytes([k])
    try:
        c = DES.new(test_key, DES.MODE_ECB)
        decrypted = c.decrypt(encrypted)
        
        if decrypted == plaintext:
            print("\nKey Found:", test_key)
            print("Decrypted:", decrypted)
            break
    except:
        continue
