from cryptography.fernet import Fernet

key = Fernet.generate_key()

with open('mykey.key', 'wb') as filekey:
    filekey.write(key)

print("Key generated and saved to mykey.key")
# last one