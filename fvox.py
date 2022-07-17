import os
import sys
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from geometry_utils import *

voxel_size = 0.5

def is_point_in_box(point, voxel):
    return check_box_point_containment(voxel, voxel_size, point)
    
def find_voxels(vertices):    
    prev = vertices[0]
    segments = [[]]
    voxels = [vertices[0]]
    seg_len = [0]    
    idx = 0
    while idx < len(vertices):
        curr = vertices[idx]

        #Go on while curr is still inside the last voxel
        if is_point_in_box(curr, voxels[-1]):
            segments[-1].append(curr)
            seg_len[-1] += np.linalg.norm(np.array(curr) - np.array(prev))
            prev = curr            
            idx += 1
            
        #Find the next voxel (probably neighbor)
        else:            
            next_v = get_next_voxel(voxels[-1], curr, voxel_size)            
            intersect1 = find_box_segment_intersection(voxels[-1], voxel_size, prev, curr)[0]
            voxels.append(next_v)
            intersect2 = find_box_segment_intersection(next_v, voxel_size, prev, curr)[0]
            #print (intersect, next_v, prev, curr)            
            seg_len[-1] += np.linalg.norm(intersect1 - prev)            
            prev = intersect2
            seg_len.append(0)
            segments.append([])

    return voxels, segments, seg_len

def get_next_voxel(prev, point, size):
    size /= 2
    x_offset = (point[0] - prev[0]) / size
    y_offset = (point[1] - prev[1]) / size
    z_offset = (point[2] - prev[2]) / size

    #print (x_offset, y_offset, z_offset)

    x_offset = math.ceil(x_offset) if x_offset >= 0 else math.floor(x_offset)
    y_offset = math.ceil(y_offset) if y_offset >= 0 else math.floor(y_offset)
    z_offset = math.ceil(z_offset) if z_offset >= 0 else math.floor(z_offset)

    x_offset = math.floor(x_offset / 2) if x_offset / 2 >= 0 else math.ceil(x_offset / 2)
    y_offset = math.floor(y_offset / 2) if y_offset / 2 >= 0 else math.ceil(y_offset / 2)
    z_offset = math.floor(z_offset / 2) if z_offset / 2 >= 0 else math.ceil(z_offset / 2)

    next_v = [prev[0] + x_offset * voxel_size, prev[1] + y_offset * voxel_size, prev[2] + z_offset * voxel_size]
    #print (prev, next_v, point, x_offset, y_offset, z_offset)
    return next_v

def filter(vox_data):
    voxels = []
    segments = []
    seg_lengths = []
    for i in range(len(vox_data[0])):
        if vox_data[2][i] >= 0.3 * voxel_size:
            voxels.append(vox_data[0][i])
            segments.append(vox_data[1][i])
            seg_lengths.append(vox_data[2][i])
    return voxels, segments, seg_lengths

#result = find_voxels([(0, 0, 0), (0, 0, 0.25), (0, 0, 0.5), (0, 0, 1.0), (0.4, 0, 1.0), (1.0, 0, 1.0)])
result = find_voxels([[ -0.9727153778076172 ,  -0.4217528998851776 ,  0.5614185929298401 ],
           [ -0.9727153182029724 ,  -0.31397244334220886 ,  1.2357196807861328 ],
           [ -0.9727153778076172 ,  -0.219588965177536 ,  1.7939447164535522 ],
           [ -0.9727153778076172 ,  -0.13024193048477173 ,  2.2572267055511475 ],
           [ -0.9727153778076172 ,  -0.03757084906101227 ,  2.6466991901397705 ],
           [ -0.9727152585983276 ,  0.06678473949432373 ,  2.9834954738616943 ],
           [ -0.9727153778076172 ,  0.19074024260044098 ,  3.287539482116699 ],
           [ -0.9727153182029724 ,  0.33555299043655396 ,  3.56065034866333 ],
           [ -0.9727153182029724 ,  0.4973544180393219 ,  3.790710687637329 ],
           [ -0.9727153182029724 ,  0.6721442937850952 ,  3.965245246887207 ],
           [ -0.9727153182029724 ,  0.8559223413467407 ,  4.071778297424316 ],
           [ -0.9727153182029724 ,  1.0446860790252686 ,  4.097873210906982 ],
           [ -0.9727153778076172 ,  1.2339951992034912 ,  4.039338111877441 ],
           [ -0.972715437412262 ,  1.4184315204620361 ,  3.910377264022827 ],
           [ -0.9727153778076172 ,  1.592444896697998 ,  3.727682590484619 ],
           [ -0.9727153182029724 ,  1.7504847049713135 ,  3.5079474449157715 ],
           [ -0.9727153778076172 ,  1.8870007991790771 ,  3.267864227294922 ],
           [ -0.9727412462234497 ,  1.9974234104156494 ,  3.022300958633423 ],
           [ -0.9730104207992554 ,  2.0844297409057617 ,  2.7726426124572754 ],
           [ -0.973825991153717 ,  2.1539461612701416 ,  2.5142297744750977 ],
           [ -0.9754915833473206 ,  2.2119131088256836 ,  2.2423744201660156 ],
           [ -0.9783106446266174 ,  2.2642719745635986 ,  1.952388882637024 ],
           [ -0.9825649261474609 ,  2.316770076751709 ,  1.640427827835083 ],
           [ -0.987775444984436 ,  2.3685264587402344 ,  1.3299603462219238 ],
           [ -0.9929015636444092 ,  2.413787364959717 ,  1.0574262142181396 ],
           [ -0.9974621534347534 ,  2.450990915298462 ,  0.8327823281288147 ],
           [ -1.0013415813446045 ,  2.480794668197632 ,  0.6524142622947693 ],
           [ -1.0045820474624634 ,  2.5045037269592285 ,  0.5086503624916077 ]])
print("Voxels: ", result[0])
print("Segments: ", result[1])
print("Segment Lengths: ", result[2])
result = filter(result)
print("Voxels: ", result[0])
print("Segments: ", result[1])
print("Segment Lengths: ", result[2])