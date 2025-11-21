"""
AES-128 with 4 rounds - wrapper for experiments
"""

from experiments.crypto.aes128_base import AES128Base


class AES128_4Round:
    """AES-128 cipher with 4 rounds."""
    
    def __init__(self):
        self._cipher = AES128Base(num_rounds=4)
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext with key."""
        return self._cipher.encrypt(plaintext, key)

