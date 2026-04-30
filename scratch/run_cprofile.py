import sys
import os
import cProfile
import pstats
import io
sys.path.insert(0, os.path.abspath('.'))

from restoration.pipeline import restore

def main():
    image_path = 'test_images/damaged_sketches/restoration_test_damaged_big.png'
    pr = cProfile.Profile()
    pr.enable()
    try:
        restore(image_path)
    except KeyboardInterrupt:
        pass
    finally:
        pr.disable()
        s = io.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(30)
        print(s.getvalue())

if __name__ == '__main__':
    main()
