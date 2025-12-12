import argparse
import json
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from pathlib import Path

def load_galaxy(checkpoint_path, mass_path):
    print(f"Loading galaxy from {checkpoint_path}...")
    vectors = np.load(checkpoint_path)
    
    print(f"Loading dictionary from {mass_path}...")
    with open(mass_path, 'r') as f:
        data = json.load(f)
        vocab = data['vocab']
        freqs = data['freq']
        
    return vectors, vocab, freqs

def main():
    parser = argparse.ArgumentParser(description="Take an HD photo of the Galaxy")
    parser.add_argument("--checkpoint", required=True, help="Path to vectors.npy")
    parser.add_argument("--mass", default="data/mass_table.json", help="Path to mass_table.json")
    parser.add_argument("--limit", type=int, default=10000, help="Top N stars to plot")
    args = parser.parse_args()

    # 1. Load Data
    vectors, vocab, freqs = load_galaxy(args.checkpoint, args.mass)
    
    # 2. Filter Top Stars
    print(f"Filtering top {args.limit} stars...")
    sub_vectors = vectors[:args.limit]
    sub_vocab = vocab[:args.limit]
    sub_freqs = freqs[:args.limit]

    # 3. Flatten Dimensions (384D -> 2D)
    print("Projecting universe to 2D (UMAP)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    projections = reducer.fit_transform(sub_vectors)

    # 4. Create Interactive Plot
    print("Developing photo...")
    df = pd.DataFrame(projections, columns=['x', 'y'])
    df['word'] = sub_vocab
    df['log_freq'] = np.log(sub_freqs) 

    fig = px.scatter(
        df, x='x', y='y', text='word', size='log_freq',
        hover_name='word', title=f"ECW-BT Galaxy Map (Top {args.limit})",
        template="plotly_dark", color='x'
    )
    fig.update_traces(textposition='top center', marker=dict(opacity=0.8))
    fig.update_layout(showlegend=False)

    output_file = "galaxy_map_hd.html"
    fig.write_html(output_file)
    print(f"Done! Open '{output_file}' in your browser.")

if __name__ == "__main__":
    main()