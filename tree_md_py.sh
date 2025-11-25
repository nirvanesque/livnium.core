#!/bin/bash
# Generate tree of .md and .py files only

find . -type f \( -name "*.md" -o -name "*.py" \) \
  -not -path "*/.venv/*" \
  -not -path "*/__pycache__/*" \
  -not -path "*/.git/*" \
  -not -path "*/site-packages/*" \
  | sed 's|^\./||' | sort | python3 - <<'EOF'
import sys

files = [line.strip() for line in sys.stdin if line.strip()]

def build_tree():
    tree = {'dirs': {}, 'files': []}
    
    for f in files:
        parts = f.split('/')
        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current['dirs']:
                current['dirs'][part] = {'dirs': {}, 'files': []}
            current = current['dirs'][part]
        current['files'].append(parts[-1])
    
    return tree

def print_tree(node, prefix='', is_last=True, is_root=True):
    items = []
    for dirname in sorted(node['dirs'].keys()):
        items.append(('dir', dirname, node['dirs'][dirname]))
    for filename in sorted(node['files']):
        items.append(('file', filename, None))
    
    for i, (item_type, name, subnode) in enumerate(items):
        is_last_item = (i == len(items) - 1)
        connector = '└── ' if is_last_item else '├── '
        print(f'{prefix}{connector}{name}')
        
        if item_type == 'dir':
            extension = '    ' if is_last_item else '│   '
            print_tree(subnode, prefix + extension, is_last_item, False)

tree = build_tree()
print_tree(tree)
EOF
