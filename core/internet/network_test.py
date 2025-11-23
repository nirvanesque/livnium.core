"""
Network Test: Entangled Basins Across Two Machines

Tests Idea A with actual network communication between two machines.
This demonstrates that two machines can achieve correlation without
sending the full basin state - only the input text and seed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import socket
import json
import pickle
import time
from typing import Optional, Dict, Any
from threading import Thread

from core.internet.entangled_basins import (
    initialize_shared_system,
    process_to_basin,
    verify_correlation,
    BasinSignatureGenerator,
    CorrelationVerifier
)


class BasinNetworkProtocol:
    """
    Simple network protocol for exchanging basin signatures.
    
    Protocol:
    - Client sends: {"seed": int, "input": str, "signature": tuple}
    - Server responds: {"correlated": bool, "match_details": dict}
    """
    
    @staticmethod
    def serialize_signature(signature: tuple) -> bytes:
        """Serialize basin signature to bytes."""
        return pickle.dumps(signature)
    
    @staticmethod
    def deserialize_signature(data: bytes) -> tuple:
        """Deserialize bytes to basin signature."""
        return pickle.loads(data)
    
    @staticmethod
    def send_signature(
        host: str,
        port: int,
        seed: int,
        input_text: str,
        signature: tuple,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Send basin signature to remote machine.
        
        Args:
            host: Remote host address
            port: Remote port
            seed: Shared seed
            input_text: Input text
            signature: Basin signature tuple
            timeout: Connection timeout
            
        Returns:
            Response dictionary with correlation results
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                
                # Send data
                message = {
                    'seed': seed,
                    'input': input_text,
                    'signature': signature
                }
                data = pickle.dumps(message)
                s.sendall(len(data).to_bytes(4, 'big'))  # Send length first
                s.sendall(data)
                
                # Receive response
                length_bytes = s.recv(4)
                if not length_bytes:
                    return {'error': 'No response received'}
                
                length = int.from_bytes(length_bytes, 'big')
                response_data = b''
                while len(response_data) < length:
                    chunk = s.recv(length - len(response_data))
                    if not chunk:
                        break
                    response_data += chunk
                
                response = pickle.loads(response_data)
                return response
                
        except socket.timeout:
            return {'error': 'Connection timeout'}
        except ConnectionRefusedError:
            return {'error': 'Connection refused - is server running?'}
        except Exception as e:
            return {'error': f'Network error: {str(e)}'}
    
    @staticmethod
    def receive_and_verify(
        port: int,
        seed: int,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Receive signature from remote machine and verify correlation.
        
        Args:
            port: Port to listen on
            seed: Shared seed
            timeout: Connection timeout
            
        Returns:
            Dictionary with correlation results
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                s.listen(1)
                
                print(f"Waiting for connection on port {port}...")
                conn, addr = s.accept()
                print(f"Connected to {addr}")
                
                with conn:
                    # Receive length
                    length_bytes = conn.recv(4)
                    if not length_bytes:
                        return {'error': 'No data received'}
                    
                    length = int.from_bytes(length_bytes, 'big')
                    
                    # Receive data
                    data = b''
                    while len(data) < length:
                        chunk = conn.recv(length - len(data))
                        if not chunk:
                            break
                        data += chunk
                    
                    message = pickle.loads(data)
                    remote_seed = message['seed']
                    remote_input = message['input']
                    remote_signature = message['signature']
                    
                    # Process same input locally
                    print(f"Received: seed={remote_seed}, input='{remote_input}'")
                    print(f"Processing locally with seed={seed}...")
                    
                    system = initialize_shared_system(seed)
                    local_signature = process_to_basin(system, remote_input, max_steps=50)
                    
                    # Verify correlation
                    result = CorrelationVerifier.verify_correlation(
                        remote_signature,
                        local_signature
                    )
                    
                    response = {
                        'correlated': result.correlated,
                        'match_details': result.match_details,
                        'local_signature_length': len(local_signature),
                        'remote_signature_length': len(remote_signature)
                    }
                    
                    # Send response
                    response_data = pickle.dumps(response)
                    conn.sendall(len(response_data).to_bytes(4, 'big'))
                    conn.sendall(response_data)
                    
                    return {
                        'success': True,
                        'correlated': result.correlated,
                        'match_details': result.match_details,
                        'local_signature': local_signature,
                        'remote_signature': remote_signature
                    }
                    
        except socket.timeout:
            return {'error': 'Connection timeout'}
        except Exception as e:
            return {'error': f'Server error: {str(e)}'}


def test_local_network_simulation():
    """
    Simulate network communication locally (for testing).
    This runs both client and server in the same process.
    """
    print("=" * 70)
    print("NETWORK TEST: Local Simulation (Two Machines in One Process)")
    print("=" * 70)
    print()
    
    seed = 42
    input_text = "hello world"
    port = 12345
    
    # Machine A: Process and get signature
    print("Machine A (Client):")
    print(f"  Initializing with seed={seed}...")
    system_a = initialize_shared_system(seed)
    print(f"  Processing input: '{input_text}'")
    signature_a = process_to_basin(system_a, input_text, max_steps=50)
    hash_a = BasinSignatureGenerator.compute_basin_hash(signature_a)
    print(f"  Basin signature: {len(signature_a)} cells")
    print(f"  Basin hash: {hash_a}")
    print()
    
    # Start server in background thread
    server_result = {'done': False, 'result': None}
    
    def server_thread():
        server_result['result'] = BasinNetworkProtocol.receive_and_verify(
            port, seed, timeout=10.0
        )
        server_result['done'] = True
    
    server = Thread(target=server_thread, daemon=True)
    server.start()
    
    # Give server time to start
    time.sleep(0.5)
    
    # Machine B: Receive and verify
    print("Machine B (Server):")
    print(f"  Waiting for signature from Machine A...")
    print()
    
    # Machine A: Send signature
    print("Machine A: Sending signature to Machine B...")
    response = BasinNetworkProtocol.send_signature(
        'localhost',
        port,
        seed,
        input_text,
        signature_a,
        timeout=10.0
    )
    
    # Wait for server to finish
    server.join(timeout=5.0)
    
    if 'error' in response:
        print(f"❌ Error: {response['error']}")
        return
    
    print()
    print("Results:")
    print(f"  Correlated: {response.get('correlated', False)}")
    print(f"  Match type: {response.get('match_details', {}).get('match_type', 'unknown')}")
    print(f"  Local signature length: {response.get('local_signature_length', 0)}")
    print(f"  Remote signature length: {response.get('remote_signature_length', 0)}")
    print()
    
    if response.get('correlated', False):
        print("✅ SUCCESS: Network correlation verified!")
        print("   Both machines reached the same basin via network communication.")
    else:
        print("⚠️  WARNING: Basins differ (may need more evolution steps)")
    print()


def test_actual_network_client(host: str, port: int, seed: int, input_text: str):
    """
    Run as client: send signature to remote server.
    
    Usage: python network_test.py client <host> <port> <seed> <input_text>
    """
    print("=" * 70)
    print("NETWORK TEST: Client Mode")
    print("=" * 70)
    print()
    
    print(f"Connecting to {host}:{port}...")
    print(f"Seed: {seed}")
    print(f"Input: '{input_text}'")
    print()
    
    # Process locally
    print("Processing input locally...")
    system = initialize_shared_system(seed)
    signature = process_to_basin(system, input_text, max_steps=50)
    hash_str = BasinSignatureGenerator.compute_basin_hash(signature)
    print(f"Basin signature: {len(signature)} cells")
    print(f"Basin hash: {hash_str}")
    print()
    
    # Send to server
    print("Sending to server...")
    response = BasinNetworkProtocol.send_signature(
        host, port, seed, input_text, signature
    )
    
    if 'error' in response:
        print(f"❌ Error: {response['error']}")
        return
    
    print()
    print("Server Response:")
    print(f"  Correlated: {response.get('correlated', False)}")
    print(f"  Match type: {response.get('match_details', {}).get('match_type', 'unknown')}")
    print()
    
    if response.get('correlated', False):
        print("✅ SUCCESS: Correlation verified across network!")
    else:
        print("⚠️  Basins differ")


def test_actual_network_server(port: int, seed: int):
    """
    Run as server: receive signature and verify.
    
    Usage: python network_test.py server <port> <seed>
    """
    print("=" * 70)
    print("NETWORK TEST: Server Mode")
    print("=" * 70)
    print()
    
    print(f"Listening on port {port}...")
    print(f"Shared seed: {seed}")
    print()
    
    result = BasinNetworkProtocol.receive_and_verify(port, seed)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    if result.get('success'):
        print()
        print("Results:")
        print(f"  Correlated: {result.get('correlated', False)}")
        print(f"  Match type: {result.get('match_details', {}).get('match_type', 'unknown')}")
        print()
        
        if result.get('correlated', False):
            print("✅ SUCCESS: Correlation verified!")
        else:
            print("⚠️  Basins differ")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "client":
            if len(sys.argv) < 6:
                print("Usage: python network_test.py client <host> <port> <seed> <input_text>")
                sys.exit(1)
            host = sys.argv[2]
            port = int(sys.argv[3])
            seed = int(sys.argv[4])
            input_text = sys.argv[5]
            test_actual_network_client(host, port, seed, input_text)
            
        elif mode == "server":
            if len(sys.argv) < 4:
                print("Usage: python network_test.py server <port> <seed>")
                sys.exit(1)
            port = int(sys.argv[2])
            seed = int(sys.argv[3])
            test_actual_network_server(port, seed)
            
        else:
            print("Unknown mode. Use 'client' or 'server'")
            sys.exit(1)
    else:
        # Default: run local simulation
        test_local_network_simulation()

