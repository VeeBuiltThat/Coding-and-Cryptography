import numpy as np
import random

# Task 1: Encoding a Message Using G
def encode_message(message, G):
    """
    Encodes a message using the generator matrix G.
    """
    codeword = np.matmul(message, G) % 2
    return codeword

G = np.array([
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1]
])

message = np.random.randint(0, 2, 3)

print("Original Information Word:", message)

codeword = encode_message(message, G)
print("Encoded Codeword:", codeword)


def introduce_error(codeword, error_position):
    """
    Flips one bit at error_position to simulate transmission errors.
    """
    received_codeword = codeword.copy()
    received_codeword[error_position] = 1 - received_codeword[error_position] 
    return received_codeword


error_position = random.randint(0, len(codeword)-1)
received_codeword = introduce_error(codeword, error_position)

print(f"Received Codeword with Error at position {error_position}:", received_codeword)


def calculate_syndrome(received_codeword, H):
    """
    Calculates the syndrome using the parity check matrix H.
    """
    syndrome = np.matmul(H, received_codeword) % 2
    error_detected = not np.array_equal(syndrome, np.zeros(H.shape[0]))
    return syndrome, error_detected

H = np.array([
    [0, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
])

syndrome, error_detected = calculate_syndrome(received_codeword, H)

print("Syndrome:", syndrome)
print("Error Detected:", error_detected)


def correct_error(received_codeword, syndrome, H):
    """
    Identifies and corrects a single-bit error using the syndrome.
    """

    for bit_pos in range(H.shape[1]):
        if np.array_equal(syndrome, H[:, bit_pos]):
            corrected = received_codeword.copy()
            corrected[bit_pos] = 1 - corrected[bit_pos] 
            return corrected
    

    return received_codeword

if error_detected:
    corrected_codeword = correct_error(received_codeword, syndrome, H)
    print("Corrected Codeword:", corrected_codeword)
else:
    print("No error correction needed.")


if error_detected and np.array_equal(corrected_codeword, codeword):
    print("Error Corrected Successfully!")
elif error_detected:
    print("Error detected but correction may not be successful.")
else:
    print("Transmission was error-free.")
