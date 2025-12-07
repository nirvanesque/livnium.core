import torch
import torch.nn.functional as F
import os
import sys

# Ensure we can import the encoder
sys.path.append(os.getcwd())
from text_encoder_quantum import QuantumTextEncoder

# CONFIG
MODEL_PATH = "model_full_physics/quantum_embeddings_final.pt"
DEVICE = "mps" 

def run_benchmark():
    print(f"💎 Loading Logic Crystal from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found.")
        return

    try:
        encoder = QuantumTextEncoder(MODEL_PATH).to(torch.device(DEVICE))
        encoder.eval()
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    # Pre-calculate normalized manifold for Cosine Search
    print("🌊 Mapping Manifold Surface...")
    all_embeddings = encoder.embed.weight.detach()
    all_embeddings = F.normalize(all_embeddings, dim=1)
    
    vocab_map = encoder.word2idx
    idx_map = encoder.idx2word

    def get_vec(word):
        return all_embeddings[vocab_map[word]] if word in vocab_map else None

    def solve(pos1, neg1, pos2):
        """
        Solves: pos1 - neg1 + pos2 = ?
        Example: King - Man + Woman = Queen
        """
        v1, v2, v3 = get_vec(pos1), get_vec(neg1), get_vec(pos2)
        
        # Check if words exist in vocab
        if v1 is None or v2 is None or v3 is None:
            missing = [w for w, v in zip([pos1, neg1, pos2], [v1, v2, v3]) if v is None]
            print(f"⚠️  Skipping {pos1}-{neg1}+{pos2}: Missing {missing}")
            return

        # --- THE PHYSICS CALCULATION ---
        # We subtract the 'concept' of neg1 and add 'concept' of pos2
        target = v1 - v2 + v3
        target = F.normalize(target, dim=0)
        
        # Search the entire vocabulary for the closest match
        scores = torch.matmul(all_embeddings, target)
        top_scores, top_idxs = torch.topk(scores, k=5)

        print(f"\n🔮 {pos1} - {neg1} + {pos2} = ?")
        found = False
        for i in range(len(top_idxs)):
            idx = top_idxs[i].item()
            word = idx_map[idx]
            
            # Skip the input words so we don't just return "King" again
            if word.lower() not in [pos1.lower(), neg1.lower(), pos2.lower()]:
                print(f"   👉 {word:<15} ({top_scores[i].item():.4f})")
                found = True
                break # Show only the top valid result
        
        if not found:
            print("   (No valid analogy found)")

    # --- THE TESTS ---
    print("\n--- 1. Gender Vector ---")
    solve("king", "man", "woman")
    solve("uncle", "man", "woman")
    
    print("\n--- 2. Geography Vector ---")
    solve("paris", "france", "germany")
    solve("tokyo", "japan", "china")
    
    print("\n--- 3. Grammar Vector ---")
    solve("cars", "car", "apple")
    solve("walking", "walk", "sing")

    print("\n--- 4. Concept Vector ---")
    solve("happy", "good", "bad")
    solve("physics", "science", "math")

if __name__ == "__main__":
    run_benchmark()