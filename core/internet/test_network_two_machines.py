"""
Test Network Communication Between Two Machines (Simulated)

This script demonstrates how to test Idea A across two machines.
It can run in server mode, client mode, or both (for local testing).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import time
import signal
import os
from core.internet.network_test import (
    BasinNetworkProtocol,
    initialize_shared_system,
    process_to_basin,
    BasinSignatureGenerator
)


def test_with_background_server():
    """
    Test network communication by starting server in background.
    """
    print("=" * 70)
    print("NETWORK TEST: Two Machines (Background Server)")
    print("=" * 70)
    print()
    
    port = 12345
    seed = 42
    input_text = "hello world"
    
    # Start server in background
    print("Starting server in background...")
    server_process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "network_test.py"),
            "server",
            str(port),
            str(seed)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give server time to start
    time.sleep(1)
    
    # Check if server is running
    if server_process.poll() is not None:
        print("❌ Server failed to start")
        stdout, stderr = server_process.communicate()
        print(f"Error: {stderr.decode()}")
        return
    
    print("✅ Server started")
    print()
    
    # Run client
    print("Running client...")
    print(f"  Seed: {seed}")
    print(f"  Input: '{input_text}'")
    print()
    
    try:
        # Process locally
        system = initialize_shared_system(seed)
        signature = process_to_basin(system, input_text, max_steps=50)
        hash_str = BasinSignatureGenerator.compute_basin_hash(signature)
        print(f"  Basin signature: {len(signature)} cells")
        print(f"  Basin hash: {hash_str}")
        print()
        
        # Send to server
        print("  Sending to server...")
        response = BasinNetworkProtocol.send_signature(
            'localhost',
            port,
            seed,
            input_text,
            signature,
            timeout=5.0
        )
        
        print()
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
        else:
            print("Server Response:")
            print(f"  Correlated: {response.get('correlated', False)}")
            print(f"  Match type: {response.get('match_details', {}).get('match_type', 'unknown')}")
            print()
            
            if response.get('correlated', False):
                print("✅ SUCCESS: Network correlation verified!")
                print("   Both machines reached the same basin via network.")
            else:
                print("⚠️  Basins differ")
        
    finally:
        # Clean up server
        print()
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✅ Server stopped")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TWO-MACHINE NETWORK TEST")
    print("=" * 70)
    print("\nThis test simulates two machines by running server in background.")
    print("For actual two-machine test, use separate terminals:\n")
    print("Terminal 1 (Server):")
    print("  python3 core/internet/network_test.py server 12345 42")
    print()
    print("Terminal 2 (Client):")
    print("  python3 core/internet/network_test.py client localhost 12345 42 \"hello world\"")
    print()
    print("=" * 70)
    print()
    
    test_with_background_server()

