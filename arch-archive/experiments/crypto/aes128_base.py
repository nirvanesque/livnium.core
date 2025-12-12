"""
AES-128 Base Implementation with Variable Rounds

Full AES-128 cipher implementation supporting variable round counts.
This is for cryptanalysis experiments, not production use.
"""

import numpy as np
from typing import List


# Standard AES S-box
AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

# Rcon for key expansion
RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]


def sub_word(word: bytes) -> bytes:
    """Substitute each byte in a word using S-box."""
    return bytes(AES_SBOX[b] for b in word)


def rot_word(word: bytes) -> bytes:
    """Rotate word left by one byte."""
    return word[1:] + word[:1]


class AES128Base:
    """
    AES-128 cipher with variable round support.
    
    This implements full AES-128 operations:
    - SubBytes, ShiftRows, MixColumns, AddRoundKey
    - Proper key expansion
    - Variable number of rounds
    """
    
    def __init__(self, num_rounds: int = 10):
        """
        Initialize AES-128 cipher.
        
        Args:
            num_rounds: Number of encryption rounds (default 10 for full AES-128)
        """
        if num_rounds < 1 or num_rounds > 10:
            raise ValueError("num_rounds must be between 1 and 10")
        self.num_rounds = num_rounds
        self.s_box = AES_SBOX
    
    def key_expansion(self, key: bytes) -> List[bytes]:
        """
        Expand 128-bit key into round keys.
        
        Args:
            key: 16-byte master key
            
        Returns:
            List of round keys (num_rounds + 1 keys, each 16 bytes)
        """
        if len(key) != 16:
            raise ValueError("AES-128 requires 16-byte key")
        
        # AES-128 key expansion produces 11 round keys (10 rounds + initial)
        # We only need num_rounds + 1 keys
        num_keys_needed = self.num_rounds + 1
        
        # Initialize with master key
        w = [key[i:i+4] for i in range(0, 16, 4)]  # Split into 4 words
        
        # Expand
        for i in range(4, 4 * num_keys_needed):
            temp = w[i - 1]
            if i % 4 == 0:
                temp = sub_word(rot_word(temp))
                # XOR first byte with Rcon
                temp_list = list(temp)
                temp_list[0] ^= RCON[i // 4]
                temp = bytes(temp_list)
            w.append(bytes(a ^ b for a, b in zip(w[i - 4], temp)))
        
        # Convert words back to 16-byte round keys
        round_keys = []
        for i in range(num_keys_needed):
            round_key = b''.join(w[4*i:4*(i+1)])
            round_keys.append(round_key)
        
        return round_keys
    
    def sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """Substitute bytes using S-box."""
        result = np.zeros_like(state)
        for i in range(len(state)):
            result[i] = self.s_box[state[i]]
        return result
    
    def shift_rows(self, state: np.ndarray) -> np.ndarray:
        """
        Shift rows operation.
        
        State is 16 bytes arranged as 4x4 matrix:
        [0  4  8  12]
        [1  5  9  13]
        [2  6  10 14]
        [3  7  11 15]
        
        Row 0: no shift
        Row 1: shift left by 1
        Row 2: shift left by 2
        Row 3: shift left by 3
        """
        if len(state) != 16:
            raise ValueError("State must be 16 bytes")
        
        result = state.copy()
        # Row 1: shift left by 1
        result[1], result[5], result[9], result[13] = result[5], result[9], result[13], result[1]
        # Row 2: shift left by 2 (swap pairs)
        result[2], result[10] = result[10], result[2]
        result[6], result[14] = result[14], result[6]
        # Row 3: shift left by 3
        result[3], result[7], result[11], result[15] = result[7], result[11], result[15], result[3]
        
        return result
    
    def mix_columns(self, state: np.ndarray) -> np.ndarray:
        """
        Mix columns operation using Galois field multiplication.
        
        Each column is multiplied by the matrix:
        [02 03 01 01]
        [01 02 03 01]
        [01 01 02 03]
        [03 01 01 02]
        """
        if len(state) != 16:
            raise ValueError("State must be 16 bytes")
        
        result = np.zeros(16, dtype=np.uint8)
        
        # Process each column (4 bytes)
        for col in range(4):
            s0 = state[col * 4 + 0]
            s1 = state[col * 4 + 1]
            s2 = state[col * 4 + 2]
            s3 = state[col * 4 + 3]
            
            # Galois field multiplication
            def gf_multiply(a, b):
                """Multiply two bytes in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1."""
                result = 0
                for _ in range(8):
                    if b & 1:
                        result ^= a
                    a <<= 1
                    if a & 0x100:
                        a ^= 0x11b  # x^8 + x^4 + x^3 + x + 1
                    b >>= 1
                return result & 0xff
            
            result[col * 4 + 0] = gf_multiply(0x02, s0) ^ gf_multiply(0x03, s1) ^ s2 ^ s3
            result[col * 4 + 1] = s0 ^ gf_multiply(0x02, s1) ^ gf_multiply(0x03, s2) ^ s3
            result[col * 4 + 2] = s0 ^ s1 ^ gf_multiply(0x02, s2) ^ gf_multiply(0x03, s3)
            result[col * 4 + 3] = gf_multiply(0x03, s0) ^ s1 ^ s2 ^ gf_multiply(0x02, s3)
        
        return result
    
    def add_round_key(self, state: np.ndarray, round_key: bytes) -> np.ndarray:
        """XOR state with round key."""
        result = state.copy()
        for i in range(min(len(state), len(round_key))):
            result[i] ^= round_key[i]
        return result
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """
        Encrypt plaintext with key.
        
        Args:
            plaintext: 16-byte plaintext
            key: 16-byte key
            
        Returns:
            16-byte ciphertext
        """
        if len(plaintext) != 16:
            raise ValueError("AES-128 requires 16-byte plaintext")
        if len(key) != 16:
            raise ValueError("AES-128 requires 16-byte key")
        
        # Generate round keys
        round_keys = self.key_expansion(key)
        
        # Convert to numpy array
        state = np.frombuffer(plaintext, dtype=np.uint8).copy()
        
        # Initial round key addition
        state = self.add_round_key(state, round_keys[0])
        
        # Main rounds
        for round_num in range(1, self.num_rounds + 1):
            state = self.sub_bytes(state)
            state = self.shift_rows(state)
            if round_num < self.num_rounds:  # No mix columns in last round
                state = self.mix_columns(state)
            state = self.add_round_key(state, round_keys[round_num])
        
        return bytes(state.tobytes())


def test_aes128():
    """Test AES-128 implementation."""
    # Test with 3 rounds
    cipher = AES128Base(num_rounds=3)
    
    key = b'\x00' * 16
    plaintext = b'\x00' * 16
    
    ciphertext = cipher.encrypt(plaintext, key)
    print(f"Key: {key.hex()}")
    print(f"Plaintext: {plaintext.hex()}")
    print(f"Ciphertext (3 rounds): {ciphertext.hex()}")
    
    # Test with different rounds
    for rounds in [2, 3, 4, 5]:
        cipher = AES128Base(num_rounds=rounds)
        ct = cipher.encrypt(plaintext, key)
        print(f"  {rounds} rounds: {ct.hex()[:32]}...")


if __name__ == "__main__":
    test_aes128()

