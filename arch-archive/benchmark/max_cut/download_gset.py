"""
Download GSET Max-Cut Benchmark Graphs

GSET is the standard benchmark suite for Max-Cut problems.
This script downloads the graphs from the official repository.
"""

import urllib.request
import urllib.error
import ssl
from pathlib import Path
import sys
from typing import List, Optional


GSET_BASE_URL = "https://web.stanford.edu/~yyye/yyye/Gset/"
GSET_GRAPHS = [
    # Small graphs for initial testing
    "G1", "G2", "G3", "G4", "G5",
    "G11", "G12", "G13", "G14", "G15",
    # Medium graphs
    "G20", "G21", "G22", "G23", "G24",
    # Large graphs (may be too large for initial testing)
    "G43", "G44", "G45", "G46", "G47", "G48", "G49", "G50",
    "G51", "G52", "G53", "G54", "G55", "G56", "G57", "G58", "G59", "G60",
    "G61", "G62", "G63", "G64", "G65", "G66", "G67", "G68", "G69", "G70",
    "G71", "G72", "G77", "G81"
]


def download_graph(graph_name: str, output_dir: Path, verbose: bool = True) -> bool:
    """
    Download a single GSET graph file.
    
    Args:
        graph_name: Name of graph (e.g., "G1")
        output_dir: Directory to save the file
        verbose: Print progress
    
    Returns:
        True if successful, False otherwise
    """
    url = f"{GSET_BASE_URL}{graph_name}"
    output_path = output_dir / graph_name
    
    if output_path.exists():
        if verbose:
            print(f"  {graph_name}: Already exists, skipping")
        return True
    
    try:
        if verbose:
            print(f"  Downloading {graph_name}...", end=" ", flush=True)
        
        # Create SSL context that doesn't verify certificates
        # (needed for some network configurations)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create request with SSL context
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_context) as response:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
        
        if verbose:
            print("✓")
        return True
    
    except urllib.error.HTTPError as e:
        if verbose:
            print(f"✗ HTTP {e.code}")
        return False
    
    except urllib.error.URLError as e:
        if verbose:
            print(f"✗ Network error: {e.reason}")
        return False
    
    except Exception as e:
        if verbose:
            print(f"✗ Error: {e}")
        return False


def download_gset_graphs(
    graph_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> int:
    """
    Download GSET graph files.
    
    Args:
        graph_names: List of graph names to download (None = all)
        output_dir: Directory to save files (default: benchmark/max_cut/gset)
        verbose: Print progress
    
    Returns:
        Number of successfully downloaded graphs
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "gset"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if graph_names is None:
        graph_names = GSET_GRAPHS
    
    if verbose:
        print(f"Downloading GSET graphs to {output_dir}")
        print(f"Total graphs: {len(graph_names)}")
        print()
    
    success_count = 0
    
    for graph_name in graph_names:
        if download_graph(graph_name, output_dir, verbose=verbose):
            success_count += 1
    
    if verbose:
        print()
        print(f"Downloaded {success_count}/{len(graph_names)} graphs")
    
    return success_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download GSET Max-Cut benchmark graphs'
    )
    parser.add_argument(
        '--graphs',
        nargs='+',
        help='Specific graphs to download (e.g., G1 G2 G3). Default: all'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: benchmark/max_cut/gset)'
    )
    parser.add_argument(
        '--small-only',
        action='store_true',
        help='Download only small graphs (G1-G5, G11-G15)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode (no progress output)'
    )
    
    args = parser.parse_args()
    
    graph_names = args.graphs
    
    if args.small_only:
        graph_names = ["G1", "G2", "G3", "G4", "G5", "G11", "G12", "G13", "G14", "G15"]
    
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    download_gset_graphs(
        graph_names=graph_names,
        output_dir=output_dir,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

