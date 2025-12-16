"""
Mindmap Ingest: Text → ThoughtNodes

Converts text files (markdown, code, plain text) into ThoughtNodes.
This is pure ingestion - no physics, no measurement, just structure.
"""

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Any, Dict
import torch


@dataclass
class ThoughtNode:
    """
    A single thought node (one paragraph).
    
    Attributes:
        id: Unique identifier (e.g., "readme_5", "arch_12")
        text: The paragraph text
        source: Source file path
        vector: Embedding vector (torch.Tensor, set after encoding)
        mass: Optional mass/importance (computed later)
    """
    id: str
    text: str
    source: str
    vector: Optional[torch.Tensor] = None
    mass: float = 1.0


# Supported file extensions
TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.rst'}
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.cc', '.cxx',
    '.h', '.hpp', '.hxx', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
    '.sh', '.bash', '.zsh', '.fish', '.yaml', '.yml', '.xml', '.html',
    '.css', '.scss', '.sass', '.less', '.sql', '.r', '.m', '.mm', '.pl', '.pm'
}
# JSON is handled specially (parsed and extracted, not treated as code)
JSON_EXTENSIONS = {'.json'}


def split_paragraphs(text: str, is_code: bool = False) -> List[str]:
    """
    Split text into paragraphs or logical blocks.
    
    Rule: One paragraph/block = one ThoughtNode.
    Not sentences. Not whole files.
    
    Args:
        text: Raw text content
        is_code: Whether this is a code file
        
    Returns:
        List of paragraph/block strings (non-empty)
    """
    if is_code:
        # For code files, try to split by function/class definitions
        # Fall back to double newlines if no clear structure
        blocks = []
        
        # Pattern for function/class definitions (works for most languages)
        # Look for common patterns: def, class, function, fn, pub fn, etc.
        pattern = r'(?:^|\n)(?:def\s+\w+|class\s+\w+|function\s+\w+|fn\s+\w+|pub\s+fn\s+\w+|func\s+\w+|public\s+\w+|private\s+\w+|protected\s+\w+|static\s+\w+)'
        
        # Find all function/class definition positions
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        
        if len(matches) > 1:
            # Split by function/class boundaries
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                block = text[start:end].strip()
                if len(block) > 20:
                    blocks.append(block)
        else:
            # Fall back to double newline splitting (works for most code)
            blocks = re.split(r'\n\n+', text)
        
        # Clean code blocks
        cleaned = []
        for block in blocks:
            block = block.strip()
            # Filter empty/minimal blocks
            if len(block) > 20:
                cleaned.append(block)
        
        return cleaned
    else:
        # For text/markdown files, split on double newlines
        paragraphs = re.split(r'\n\n+', text)
        
        # Clean and filter
        cleaned = []
        for para in paragraphs:
            # Remove markdown headers (keep content)
            para = re.sub(r'^#+\s+', '', para, flags=re.MULTILINE)
            # Remove code blocks (keep the text, not the code)
            para = re.sub(r'```[\s\S]*?```', '', para)
            # Remove inline code markers
            para = re.sub(r'`[^`]+`', '', para)
            # Strip whitespace
            para = para.strip()
            # Filter empty/minimal paragraphs
            if len(para) > 20:  # Minimum meaningful length
                cleaned.append(para)
        
        return cleaned


def extract_json_content(data: Any, path: str = "", max_depth: int = 10, current_depth: int = 0) -> List[str]:
    """
    Extract meaningful text content from JSON structure.
    
    Args:
        data: JSON data (dict, list, or primitive)
        path: Current path in JSON structure
        max_depth: Maximum recursion depth to prevent stack overflow
        current_depth: Current recursion depth
        
    Returns:
        List of text strings extracted from JSON
    """
    if current_depth >= max_depth:
        return []  # Prevent infinite recursion
    
    texts = []
    
    if isinstance(data, dict):
        # For dicts, create key-value pairs
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, str) and len(value) > 5:
                # String values are meaningful
                texts.append(f"{key}: {value}")
            elif isinstance(value, (int, float, bool)):
                texts.append(f"{key}: {value}")
            elif isinstance(value, (dict, list)):
                # Recurse into nested structures
                texts.extend(extract_json_content(value, current_path, max_depth, current_depth + 1))
    elif isinstance(data, list):
        # For lists, extract items (limit to first 1000 items to prevent explosion)
        limit = 1000 if current_depth == 0 else 100
        for i, item in enumerate(data[:limit]):
            current_path = f"{path}[{i}]" if path else f"[{i}]"
            if isinstance(item, str) and len(item) > 5:
                texts.append(item)
            elif isinstance(item, (int, float, bool)):
                texts.append(str(item))
            elif isinstance(item, (dict, list)):
                texts.extend(extract_json_content(item, current_path, max_depth, current_depth + 1))
    elif isinstance(data, str) and len(data) > 5:
        texts.append(data)
    elif isinstance(data, (int, float, bool)):
        texts.append(str(data))
    
    return texts


def ingest_file(file_path: Path, prefix: Optional[str] = None) -> List[ThoughtNode]:
    """
    Ingest any text/code/JSON file into ThoughtNodes.
    
    Args:
        file_path: Path to file (txt, md, code, json, etc.)
        prefix: Optional prefix for node IDs (defaults to filename stem)
        
    Returns:
        List of ThoughtNodes (one per paragraph/block)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle JSON files specially
    if file_path.suffix.lower() == '.json':
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # For very large JSON files (>50MB), use sampling/limiting
        if file_size_mb > 50:
            print(f"  ⚠ Large JSON file ({file_size_mb:.1f}MB) - using sampling")
            max_nodes = 10000  # Limit to 10K nodes for large files
        else:
            max_nodes = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract text content from JSON structure
            json_texts = extract_json_content(json_data, max_depth=10)
            
            # Apply sampling for large files
            if max_nodes and len(json_texts) > max_nodes:
                import random
                total_count = len(json_texts)
                json_texts = random.sample(json_texts, max_nodes)
                print(f"  → Sampled {max_nodes} items from {total_count} total")
            
            if not json_texts:
                # If no meaningful text extracted, use JSON string representation
                json_str = json.dumps(json_data, indent=2)
                paragraphs = split_paragraphs(json_str, is_code=False)
            else:
                # Combine into paragraphs (group related items)
                paragraphs = []
                current_para = []
                for text in json_texts:
                    if len(text.strip()) > 5:  # Meaningful content
                        current_para.append(text.strip())
                        # Group every 2-4 items or if text is long
                        if len(current_para) >= 3 or len(text) > 100:
                            paragraphs.append(" | ".join(current_para))
                            current_para = []
                if current_para:
                    paragraphs.append(" | ".join(current_para))
                
                # Filter minimal paragraphs
                paragraphs = [p for p in paragraphs if len(p) > 20]
            
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as text
            try:
                text = file_path.read_text(encoding='utf-8')
                paragraphs = split_paragraphs(text, is_code=False)
            except:
                print(f"Warning: Could not read {file_path}, skipping")
                return []
        except Exception as e:
            print(f"Warning: Error processing JSON {file_path}: {e}, skipping")
            return []
    else:
        # Check if it's a code file
        is_code = file_path.suffix.lower() in CODE_EXTENSIONS
        
        # Try to read file (handle encoding errors)
        try:
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try latin-1 as fallback
            try:
                text = file_path.read_text(encoding='latin-1')
            except:
                print(f"Warning: Could not read {file_path}, skipping")
                return []
        
        # Split into paragraphs/blocks
        paragraphs = split_paragraphs(text, is_code=is_code)
    
    # Generate prefix if not provided
    if prefix is None:
        prefix = file_path.stem.lower().replace('-', '_').replace(' ', '_')
        # Add parent directory if in sources folder for uniqueness
        if 'sources' in str(file_path):
            parent = file_path.parent.name
            if parent != 'sources':
                prefix = f"{parent}_{prefix}"
    
    # Create ThoughtNodes
    nodes = []
    for i, para in enumerate(paragraphs):
        node_id = f"{prefix}_{i}"
        node = ThoughtNode(
            id=node_id,
            text=para,
            source=str(file_path)
        )
        nodes.append(node)
    
    return nodes


def ingest_markdown(file_path: Path, prefix: Optional[str] = None) -> List[ThoughtNode]:
    """
    Legacy function for backward compatibility.
    Use ingest_file() instead.
    """
    return ingest_file(file_path, prefix)


def scan_sources_folder(sources_dir: Optional[Path] = None) -> List[Path]:
    """
    Scan the sources folder for all supported text/code files.
    
    Args:
        sources_dir: Path to sources folder (defaults to mindmap/sources/)
        
    Returns:
        List of file paths to ingest
    """
    if sources_dir is None:
        # Default to mindmap/sources/ relative to this file
        sources_dir = Path(__file__).parent / "sources"
    
    if not sources_dir.exists():
        return []
    
    # Collect all supported files
    all_extensions = TEXT_EXTENSIONS | CODE_EXTENSIONS | JSON_EXTENSIONS
    files = []
    
    # Scan recursively
    for ext in all_extensions:
        files.extend(sources_dir.rglob(f"*{ext}"))
    
    # Filter out common ignore patterns
    ignore_patterns = {
        '__pycache__', '.git', '.svn', 'node_modules', '.venv', 'venv',
        'env', '.env', 'build', 'dist', '.pytest_cache', '.mypy_cache'
    }
    
    filtered = []
    for file_path in files:
        # Check if any ignore pattern is in the path
        parts = file_path.parts
        if not any(ignore in parts for ignore in ignore_patterns):
            filtered.append(file_path)
    
    return sorted(filtered)


def ingest_sources_folder(sources_dir: Optional[Path] = None) -> List[ThoughtNode]:
    """
    Ingest all files from the sources folder.
    
    Args:
        sources_dir: Path to sources folder (defaults to mindmap/sources/)
        
    Returns:
        List of all ThoughtNodes from all files
    """
    files = scan_sources_folder(sources_dir)
    
    try:
        from tqdm import tqdm
        file_iterator = tqdm(files, desc="Ingesting files", unit="file")
    except ImportError:
        file_iterator = files
    
    all_nodes = []
    for file_path in file_iterator:
        try:
            nodes = ingest_file(file_path)
            all_nodes.extend(nodes)
            if not hasattr(file_iterator, 'set_postfix'):
                print(f"  → {file_path.name}: {len(nodes)} nodes")
            else:
                file_iterator.set_postfix_str(f"{file_path.name}: {len(nodes)} nodes")
        except Exception as e:
            print(f"  ⚠ Error ingesting {file_path}: {e}")
    
    return all_nodes


def embed_thoughts(nodes: List[ThoughtNode], encoder, show_progress: bool = True) -> None:
    """
    Embed ThoughtNodes using provided encoder.
    
    This happens outside the kernel - it's domain-level encoding.
    The encoder should have an encode() method that takes text and returns a tensor.
    
    Args:
        nodes: List of ThoughtNodes to embed
        encoder: Encoder with encode(text: str) -> torch.Tensor method
        show_progress: Whether to show progress bar
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(nodes, desc="Embedding", unit="nodes") if show_progress else nodes
    except ImportError:
        iterator = nodes
    
    for node in iterator:
        node.vector = encoder.encode(node.text)

