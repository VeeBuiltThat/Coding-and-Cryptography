from cryptography.fernet import Fernet

with open('mykey.key', 'rb') as filekey:
    key = filekey.read()

fernet = Fernet(key)

with open('my_details.txt', 'rb') as file:
    encrypted_data = file.read()

decrypted_data = fernet.decrypt(encrypted_data)

with open('my_details.txt', 'wb') as file:
    file.write(decrypted_data)

print("File successfully decrypted!")
