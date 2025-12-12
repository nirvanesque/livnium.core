"""
Simplified AES-32: Reduced version for testing cryptanalysis

AES-32 uses:
- 32-bit key (4 bytes)
- 32-bit block (4 bytes)
- 4 rounds (simplified from 10)
- Simplified S-box and operations
"""

import numpy as np
from typing import List, Tuple


class AES32:
    """
    Simplified AES-32 cipher for cryptanalysis testing.
    
    This is NOT cryptographically secure - it's a simplified version
    for testing geometric cryptanalysis approaches.
    """
    
    def __init__(self):
        """Initialize AES-32 with simplified S-box."""
        # Simplified 4x4 S-box (16 values)
        self.s_box = np.array([
            0x9, 0x4, 0xa, 0xb,
            0xd, 0x1, 0x8, 0x5,
            0x6, 0x2, 0x0, 0x3,
            0xc, 0xe, 0xf, 0x7
        ], dtype=np.uint8)
        
        # Inverse S-box
        self.inv_s_box = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            self.inv_s_box[self.s_box[i]] = i
    
    def key_schedule(self, key: bytes) -> List[bytes]:
        """
        Generate round keys from 32-bit master key.
        
        Args:
            key: 32-bit key (4 bytes)
            
        Returns:
            List of round keys (5 keys for 4 rounds + initial)
        """
        if len(key) != 4:
            raise ValueError("AES-32 requires 4-byte key")
        
        round_keys = [key]
        
        # Simplified key expansion
        for round_num in range(4):
            prev_key = round_keys[-1]
            # Rotate and substitute
            new_key = bytearray(4)
            new_key[0] = self.s_box[prev_key[3] & 0xF] ^ prev_key[0] ^ round_num
            new_key[1] = prev_key[0] ^ prev_key[1]
            new_key[2] = prev_key[1] ^ prev_key[2]
            new_key[3] = prev_key[2] ^ prev_key[3]
            round_keys.append(bytes(new_key))
        
        return round_keys
    
    def sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """Substitute bytes using S-box."""
        result = np.zeros_like(state)
        for i in range(len(state)):
            result[i] = self.s_box[state[i] & 0xF]
        return result
    
    def shift_rows(self, state: np.ndarray) -> np.ndarray:
        """
        Shift rows (simplified for 4-byte state).
        
        For 4 bytes arranged as 2x2:
        [a b]  ->  [a b]
        [c d]      [d c]  (swap bottom row)
        """
        if len(state) != 4:
            raise ValueError("State must be 4 bytes")
        
        result = state.copy()
        # Swap bytes 2 and 3
        result[2], result[3] = result[3], result[2]
        return result
    
    def mix_columns(self, state: np.ndarray) -> np.ndarray:
        """
        Mix columns (simplified for 4-byte state).
        
        Simplified mixing: XOR with rotated state
        """
        result = state.copy()
        # Simple mixing: XOR with rotated
        result[0] ^= result[1]
        result[1] ^= result[2]
        result[2] ^= result[3]
        result[3] ^= result[0]
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
            plaintext: 4-byte plaintext
            key: 4-byte key
            
        Returns:
            4-byte ciphertext
        """
        if len(plaintext) != 4:
            raise ValueError("AES-32 requires 4-byte plaintext")
        if len(key) != 4:
            raise ValueError("AES-32 requires 4-byte key")
        
        # Generate round keys
        round_keys = self.key_schedule(key)
        
        # Convert to numpy array
        state = np.frombuffer(plaintext, dtype=np.uint8)
        
        # Initial key addition
        state = self.add_round_key(state, round_keys[0])
        
        # 4 rounds
        for round_num in range(1, 5):
            state = self.sub_bytes(state)
            state = self.shift_rows(state)
            if round_num < 4:  # No mix columns in last round
                state = self.mix_columns(state)
            state = self.add_round_key(state, round_keys[round_num])
        
        return bytes(state.tobytes())
    
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """
        Decrypt ciphertext with key.
        
        Args:
            ciphertext: 4-byte ciphertext
            key: 4-byte key
            
        Returns:
            4-byte plaintext
        """
        if len(ciphertext) != 4:
            raise ValueError("AES-32 requires 4-byte ciphertext")
        if len(key) != 4:
            raise ValueError("AES-32 requires 4-byte key")
        
        # Generate round keys
        round_keys = self.key_schedule(key)
        
        # Convert to numpy array
        state = np.frombuffer(ciphertext, dtype=np.uint8)
        
        # Reverse rounds (inverse order)
        # Start with last round key
        for round_num in range(4, 0, -1):
            state = self.add_round_key(state, round_keys[round_num])
            if round_num < 4:
                # Inverse mix columns (self-inverse for XOR operations)
                state = self.mix_columns_inv(state)
            state = self.shift_rows(state)  # Shift rows is self-inverse
            state = self.sub_bytes_inv(state)
        
        # Initial key addition
        state = self.add_round_key(state, round_keys[0])
        
        return bytes(state.tobytes())
    
    def mix_columns_inv(self, state: np.ndarray) -> np.ndarray:
        """
        Inverse mix columns (for decryption).
        
        Since mix_columns uses XOR, the inverse is the same operation.
        """
        return self.mix_columns(state)
    
    def sub_bytes_inv(self, state: np.ndarray) -> np.ndarray:
        """Inverse substitute bytes using inverse S-box."""
        result = np.zeros_like(state)
        for i in range(len(state)):
            result[i] = self.inv_s_box[state[i] & 0xF]
        return result


def test_aes32():
    """Test AES-32 implementation."""
    cipher = AES32()
    
    # Test encryption
    key = b'\x12\x34\x56\x78'
    plaintext = b'\x9a\xbc\xde\xf0'
    
    ciphertext = cipher.encrypt(plaintext, key)
    print(f"Key: {key.hex()}")
    print(f"Plaintext: {plaintext.hex()}")
    print(f"Ciphertext: {ciphertext.hex()}")
    
    # Test decryption
    decrypted = cipher.decrypt(ciphertext, key)
    print(f"Decrypted: {decrypted.hex()}")
    print(f"Match: {plaintext == decrypted}")


if __name__ == "__main__":
    test_aes32()

