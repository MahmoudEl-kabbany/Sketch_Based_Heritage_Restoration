from pygeoops import centerline
from shapely.geometry import Polygon
import numpy as np

# A rectangle that represents a thick stroke
poly = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])

# default
cl_default = centerline(poly, densify_distance=-1.0, min_branch_length=0.0)
print("Default length:", cl_default.length if cl_default else 0)

# With min branch
cl_pruned = centerline(poly, densify_distance=-1.0, min_branch_length=15.0)
print("Pruned length:", cl_pruned.length if cl_pruned else 0)

# A star shape
star = Polygon([(50, 100), (60, 60), (100, 50), (60, 40), (50, 0), (40, 40), (0, 50), (40, 60)])
cl_star_default = centerline(star, densify_distance=-1.0, min_branch_length=0.0)
print("Star default length:", cl_star_default.length if cl_star_default else 0)

cl_star_pruned = centerline(star, densify_distance=-1.0, min_branch_length=15.0)
print("Star pruned length:", cl_star_pruned.length if cl_star_pruned else 0)
