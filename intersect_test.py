import os
import sys
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from geometry_utils import *


intersect = find_box_segment_intersection([-0.9727153778076172, -0.4217528998851776, -0.4385814070701599], 1.0, [ -0.9727153778076172 ,  -0.4217528998851776 ,  0.5614185929298401 ],\
           [ -0.9727153182029724 ,  -0.31397244334220886 ,  1.2357196807861328 ])

print (intersect)