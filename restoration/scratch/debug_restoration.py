
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from restoration.extraction import extract_paths
from restoration.candidates import generate_candidates
from restoration.synthesis import synthesize_bridges

def debug_image(image_name):
    print(f"\n--- Debugging {image_name} ---")
    image_path = os.path.join("test_images/damaged_sketches", image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    result = extract_paths(image_path)
    candidates = generate_candidates(result)
    
    # We want to find specific candidates mentioned by the user
    # Star Junction: R3, R4 at the center
    # Ankh: R1 at the top loop
    
    for i, c in enumerate(candidates):
        # The labeling in the overlay might match the order here or be different
        # Usually R1, R2... in overlay are the first N accepted candidates.
        # But let's just look at all candidates.
        
        ep_a = c.ep_a
        ep_b = c.ep_b
        
        dist = np.linalg.norm(ep_a.position - ep_b.position)
        
        print(f"Candidate {i}: dist={dist:.2f}, scenario={c.scenario}")
        print(f"  EP A: pos={ep_a.position}, tang={ep_a.tangent}, curv={ep_a.curvature:.6f}, conf={ep_a.tangent_confidence:.2f}")
        print(f"  EP B: pos={ep_b.position}, tang={ep_b.tangent}, curv={ep_b.curvature:.6f}, conf={ep_b.tangent_confidence:.2f}")

if __name__ == "__main__":
    debug_image("star_junction.png")
    debug_image("damaged_ankh.png")
