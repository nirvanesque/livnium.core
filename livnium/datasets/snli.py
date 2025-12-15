"""
SNLI Dataset Loader

Loads SNLI (Stanford Natural Language Inference) dataset.
"""

import json
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from livnium.datasets.base import LivniumDataset


class SNLIDataset(LivniumDataset):
    """
    SNLI dataset loader.
    
    Loads SNLI data from JSONL files.
    """
    
    LABEL_MAP = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }
    
    def __init__(
        self,
        data_path: str,
        vocab_size: int = 1000,
        max_length: int = 50,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize SNLI dataset.
        
        Args:
            data_path: Path to SNLI JSONL file
            vocab_size: Vocabulary size for simple tokenizer
            max_length: Maximum sequence length
            tokenizer: Optional tokenizer (if None, uses simple word-based)
        """
        self.data_path = Path(data_path)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Load data
        self.samples = []
        if self.data_path.exists():
            with open(self.data_path, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    if sample.get("gold_label") in self.LABEL_MAP:
                        self.samples.append(sample)
        else:
            # Create dummy data if file doesn't exist
            print(f"Warning: {data_path} not found, creating dummy data")
            self.samples = self._create_dummy_data(100)
        
        # Build vocabulary if needed
        if self.tokenizer is None:
            self.vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build simple vocabulary from data."""
        vocab = {"<pad>": 0, "<unk>": 1}
        word_counts = {}
        
        for sample in self.samples:
            for text in [sample.get("sentence1", ""), sample.get("sentence2", "")]:
                for word in text.lower().split():
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Take top vocab_size words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 2]:
            vocab[word] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization."""
        if self.tokenizer is not None:
            return self.tokenizer(text)
        
        tokens = [self.vocab.get(word.lower(), self.vocab["<unk>"]) 
                 for word in text.split()]
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab["<pad>"]] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def _create_dummy_data(self, size: int) -> List[Dict]:
        """Create dummy data for testing."""
        return [
            {
                "sentence1": f"premise {i}",
                "sentence2": f"hypothesis {i}",
                "gold_label": ["entailment", "neutral", "contradiction"][i % 3]
            }
            for i in range(size)
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        prem_ids = torch.tensor(self._tokenize(sample.get("sentence1", "")), dtype=torch.long)
        hyp_ids = torch.tensor(self._tokenize(sample.get("sentence2", "")), dtype=torch.long)
        label = self.LABEL_MAP.get(sample.get("gold_label", "neutral"), 1)
        
        return {
            "premise_ids": prem_ids,
            "hypothesis_ids": hyp_ids,
            "label": label,
        }

