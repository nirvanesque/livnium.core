"""
AES-128 with 5 rounds - wrapper for experiments
"""

from experiments.crypto.aes128_base import AES128Base


class AES128_5Round:
    """AES-128 cipher with 5 rounds."""
    
    def __init__(self):
        self._cipher = AES128Base(num_rounds=5)
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext with key."""
        return self._cipher.encrypt(plaintext, key)

