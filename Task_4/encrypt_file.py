from cryptography.fernet import Fernet

with open('mykey.key', 'rb') as filekey:
    key = filekey.read()

fernet = Fernet(key)

with open('my_details.txt', 'rb') as file:
    original_data = file.read()

encrypted_data = fernet.encrypt(original_data)

with open('my_details.txt', 'wb') as file:
    file.write(encrypted_data)

print("File successfully encrypted!")
# this the second