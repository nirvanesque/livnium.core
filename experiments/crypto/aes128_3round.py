"""
AES-128 with 3 rounds - wrapper for experiments
"""

from experiments.crypto.aes128_base import AES128Base


class AES128_3Round:
    """AES-128 cipher with 3 rounds."""
    
    def __init__(self):
        self._cipher = AES128Base(num_rounds=3)
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext with key."""
        return self._cipher.encrypt(plaintext, key)

