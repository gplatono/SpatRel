import os
import sys
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from geometry_utils import *


intersect = find_box_segment_intersection([1, 1, 1], 1.0,  [1, 1.4, 1.4],\
            [1, 2, 2])
#intersect = find_point_plane_projection((2, 2, 2), [(1,0,0), (0, 1, 0), (0, 0, 1)])

print (intersect)