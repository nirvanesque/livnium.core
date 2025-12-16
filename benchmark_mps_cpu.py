#!/usr/bin/env python3
"""
Benchmark MPS vs CPU for SNLI training
"""
import torch
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from livnium.examples.train_snli import SNLIModel
from livnium.datasets.snli import SNLIDataset
from torch.utils.data import DataLoader

def benchmark_device(device: str, num_batches: int = 50, batch_size: int = 32):
    """Benchmark training speed on a specific device."""
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device.upper()}")
    print(f"{'='*60}")
    
    # Create model
    model = SNLIModel(
        dim=256,
        num_layers=5,
        vocab_size=2000,
        enable_basins=False,  # Disable basins for fair comparison
    ).to(device)
    
    # Create dummy dataset
    vocab_size = 2000
    dummy_data = []
    for i in range(batch_size * num_batches):
        dummy_data.append({
            "premise_ids": torch.randint(0, vocab_size, (50,)),
            "hypothesis_ids": torch.randint(0, vocab_size, (50,)),
            "label": torch.randint(0, 3, (1,))[0]  # Scalar label
        })
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Warmup (MPS needs warmup)
    print("Warming up...")
    model.train()
    for _ in range(5):
        batch = dummy_data[0]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Ensure label is a tensor
        if not isinstance(batch["label"], torch.Tensor):
            batch["label"] = torch.tensor(batch["label"], device=device)
        logits = model(batch)
        label_tensor = batch["label"] if batch["label"].dim() > 0 else batch["label"].unsqueeze(0)
        loss = loss_fn(logits, label_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Synchronize for accurate timing
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {num_batches} batches...")
    start_time = time.time()
    
    for i in range(num_batches):
        batch = dummy_data[i % len(dummy_data)]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Ensure label is a tensor
        if not isinstance(batch["label"], torch.Tensor):
            batch["label"] = torch.tensor(batch["label"], device=device)
        
        optimizer.zero_grad()
        logits = model(batch)
        label_tensor = batch["label"] if batch["label"].dim() > 0 else batch["label"].unsqueeze(0)
        loss = loss_fn(logits, label_tensor)
        loss.backward()
        optimizer.step()
    
    # Synchronize again
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    batches_per_sec = num_batches / elapsed
    samples_per_sec = batches_per_sec * batch_size
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Batches/sec: {batches_per_sec:.2f}")
    print(f"Samples/sec: {samples_per_sec:.2f}")
    
    return elapsed, batches_per_sec, samples_per_sec

if __name__ == "__main__":
    print("SNLI Training Speed Benchmark: MPS vs CPU")
    print("=" * 60)
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    print(f"MPS available: {mps_available}")
    
    # Benchmark CPU
    cpu_time, cpu_bps, cpu_sps = benchmark_device("cpu", num_batches=50, batch_size=32)
    
    # Benchmark MPS if available (try larger batch for MPS efficiency)
    if mps_available:
        print("\nTesting MPS with larger batch size (MPS needs larger batches)...")
        mps_time_large, mps_bps_large, mps_sps_large = benchmark_device("mps", num_batches=30, batch_size=128)
        mps_time, mps_bps, mps_sps = benchmark_device("mps", num_batches=50, batch_size=32)
        
        # Compare
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        speedup = cpu_time / mps_time
        print(f"CPU time (batch=32): {cpu_time:.2f}s")
        print(f"MPS time (batch=32): {mps_time:.2f}s")
        print(f"MPS time (batch=128): {mps_time_large:.2f}s")
        print(f"Speedup (batch=32): {speedup:.2f}x")
        print(f"CPU samples/sec: {cpu_sps:.2f}")
        print(f"MPS samples/sec (batch=32): {mps_sps:.2f}")
        print(f"MPS samples/sec (batch=128): {mps_sps_large:.2f}")
        
        # Compare with larger batch
        cpu_time_large = cpu_time * (128/32)  # Estimate CPU time for larger batch
        speedup_large = cpu_time_large / mps_time_large if mps_time_large > 0 else 0
        
        if speedup < 1.0:
            print(f"\n⚠️  WARNING: MPS is {1/speedup:.2f}x SLOWER than CPU with batch=32!")
            if speedup_large > 1.0:
                print(f"✓ But MPS is {speedup_large:.2f}x FASTER with batch=128!")
                print("  → Use larger batch sizes for MPS training")
            else:
                print("This indicates a performance issue.")
        elif speedup < 1.5:
            print(f"\n⚠️  WARNING: MPS speedup is minimal ({speedup:.2f}x) with batch=32")
            if speedup_large > 1.5:
                print(f"✓ But MPS is {speedup_large:.2f}x faster with batch=128!")
                print("  → Use larger batch sizes for MPS training")
            else:
                print("MPS may not be providing expected acceleration.")
        else:
            print(f"\n✓ MPS is {speedup:.2f}x faster than CPU")
    else:
        print("\nMPS not available, skipping MPS benchmark")

