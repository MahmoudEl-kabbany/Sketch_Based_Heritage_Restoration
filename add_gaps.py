import cv2
import numpy as np
import random
import argparse
import os

def apply_gaps(img, num_gaps=20, gap_size_min=5, gap_size_max=15, gap_type='circle', threshold=127):
    """
    Applies gaps to lines in a black and white image array.
    Lines are assumed to be black (0) and background white (255).
    
    Args:
        img: Numpy array (grayscale).
        num_gaps: Number of gaps to add.
        gap_size_min/max: Size range for gaps.
        gap_type: 'circle', 'crack', 'scrub', 'box'.
        threshold: Threshold for identifying black lines.
        
    Returns:
        Numpy array with gaps applied.
    """
    # Binary thresholding to ensure clear black/white
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Find coordinates of all black pixels (lines)
    black_pixels = np.column_stack(np.where(binary == 0))
    
    if len(black_pixels) == 0:
        return img.copy()

    # Work on a copy
    result = img.copy()
    
    # Sample gap locations
    effective_num_gaps = min(num_gaps, len(black_pixels))
    indices = random.sample(range(len(black_pixels)), effective_num_gaps)
    gap_seeds = black_pixels[indices]

    for y, x in gap_seeds:
        size = random.randint(gap_size_min, gap_size_max)
        
        if gap_type == 'circle':
            cv2.circle(result, (x, y), size, 255, -1)
        
        elif gap_type == 'crack':
            angle = random.uniform(0, 180)
            length = size * 2
            thickness = max(1, size // 4)
            dx = int(np.cos(np.radians(angle)) * length / 2)
            dy = int(np.sin(np.radians(angle)) * length / 2)
            cv2.line(result, (x - dx, y - dy), (x + dx, y + dy), 255, thickness)
            
        elif gap_type == 'scrub':
            for _ in range(size * 10):
                nx = x + random.randint(-size, size)
                ny = y + random.randint(-size, size)
                if 0 <= nx < result.shape[1] and 0 <= ny < result.shape[0]:
                    result[ny, nx] = 255
        
        elif gap_type == 'box':
            cv2.rectangle(result, (x - size, y - size), (x + size, y + size), 255, -1)

    return result

def add_gaps_file(input_path, output_path, **kwargs):
    """Wrapper to process a single file."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    result = apply_gaps(img, **kwargs)
    cv2.imwrite(output_path, result)
    return result

def add_gaps_batch(input_paths, output_dir, **kwargs):
    """Processes multiple image paths and saves them to an output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    for path in input_paths:
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        try:
            res = add_gaps_file(path, out_path, **kwargs)
            results.append(res)
            print(f"Processed {filename} -> {out_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Add gaps to black and white sketch images.")
    parser.add_argument("input", help="Path to input file or directory.")
    parser.add_argument("output", help="Path to output file or directory.")
    parser.add_argument("--num", type=int, default=20, help="Number of gaps.")
    parser.add_argument("--min_size", type=int, default=5, help="Min gap size.")
    parser.add_argument("--max_size", type=int, default=15, help="Max gap size.")
    parser.add_argument("--type", choices=['circle', 'crack', 'scrub', 'box'], default='circle', help="Gap type.")
    parser.add_argument("--threshold", type=int, default=127, help="Black detection threshold.")
    parser.add_argument("--seed", type=int, help="Random seed.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Process either single file or directory
    if os.path.isdir(args.input):
        extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(extensions)]
        add_gaps_batch(paths, args.output, num_gaps=args.num, gap_size_min=args.min_size, 
                       gap_size_max=args.max_size, gap_type=args.type, threshold=args.threshold)
    else:
        add_gaps_file(args.input, args.output, num_gaps=args.num, gap_size_min=args.min_size, 
                      gap_size_max=args.max_size, gap_type=args.type, threshold=args.threshold)

if __name__ == "__main__":
    main()
