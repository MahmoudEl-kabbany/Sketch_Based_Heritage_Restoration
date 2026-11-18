import glob, sys
from restoration.pipeline import restore_batch

if __name__ == "__main__":
    folder = sys.argv[1]
    images = sorted(glob.glob(f"{folder}/*.png") + glob.glob(f"{folder}/*.jpg"))
    restore_batch(images, workers=4, output_dir="results/new")
