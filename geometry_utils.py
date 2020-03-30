import math
import numpy as np
import bpy, bmesh
from mathutils.bvhtree import BVHTree


#Computes the value of the univariate Gaussian
#Inputs: x - random variable value; mu - mean; sigma - variance
#Return value: real number
def gaussian(x, mu, sigma):
    return math.e ** (- 0.5 * ((float(x) - mu) / sigma) ** 2) / (math.fabs(sigma) * math.sqrt(2.0 * math.pi))

#Computes the value of the logistic sigmoid function
#Inputs: x - random variable value; a, b - coefficients
#Return value: real number
def sigmoid(x, a, b):
    return a / (1 + math.e ** (- b * x)) if b * x > -100 else 0


#Computes the cross-product of vectors a and b
#Inputs: a,b - vector coordinates as tuples or lists
#Return value: a triple of coordinates
def cross_product(a, b):
    return (a[1] * b[2] - a[2] * b[1], b[0] * a[2] - b[2] * a[0],
            a[0] * b[1] - a[1] * b[0])

#Given three points that define a plane, computes the unit normal vector to that plane
#Inputs: a,b,c - point coordinates as tuples or lists
#Return value: normal vector as a triple of coordinates
def get_normal(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    u = b - a
    v = c - a
    u_x_v = np.cross(u, v)    
    return u_x_v / np.linalg.norm(u_x_v)
    #return cross_product((a[0] - b[0], a[1] - b[1], a[2] - b[2]),
#                         (c[0] - b[0], c[1] - b[1], c[2] - b[2]))

#Given a point and a plane defined by a, b and c
#computes the orthogonal distance from the point to that plane
#Inputs: point,a,b,c - point coordinates as tuples or lists
#Return value: real number
def get_distance_from_plane(point, a, b, c):
    normal = np.array(get_normal(a, b, c))
    return math.fabs((np.array(point).dot(normal) - np.array(a).dot(normal)) / np.linalg.norm(normal))

#Computes the orthogonal distance between x3 and the line
#defined by x1 and x2
#Inputs: x1, x2, x3 - point coordinates as tuples or lists
#Return value: real number
def get_distance_from_line(x1, x2, x3):
    if np.linalg.norm(x1 - x2) <= 0.001:
        return point_distance(x1, x3)    
    #print ("POINTS: {}, {}, {}".format(x1, x2, x3))
    v1 = np.array(x3) - np.array(x1)
    v2 = np.array(x2) - np.array(x1)
    #v1 = np.array(x3 - x1)
    #v2 = np.array(x2 - x1)
    #print ("VECTORS: {}, {}".format(v1, v2))
    l1 = np.linalg.norm(v1)
    l2 = np.dot(v1, v2) / np.linalg.norm(v2)
    #print ("L1, L2", l1, l2)
    return math.sqrt(math.fabs(l1 * l1 - l2 * l2))
    #t = (x3[0] - x1[0]) * (x2[0] - x1[0]) + (x3[1] - x1[1]) * (x2[1] - x1[1]) * (x3[2] - x1[2]) * (x2[2] - x1[2])
    #dist = point_distance(x1, x2) ** 2    
    #t = t / dist if dist != 0 else 1e10
    #x0 = (x1[0] + (x2[0] - x1[0]) * t, x1[1] + (x2[1] - x1[1]) * t, x1[2] + (x2[2] - x1[2]) * t)
    #return point_distance(x0, x3)

#Computes a simple Euclidean distance between two points
#Inputs: a, b - point coordinates as tuples or lists
#Return value: real number
def point_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


#Computes the projection of the bounding box of a set
#of points onto the XY-plane
#Inputs: a, b - point coordinates as tuples or lists
#Return value: real number
def get_2d_bbox(points):
    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9    
    for p in points:
        min_x = min(min_x, p[0])
        min_y = min(min_y, p[1])
        max_x = max(max_x, p[0])
        max_y = max(max_y, p[1])       
    return [min_x, max_x, min_y, max_y]

#Computes the distance between the centroids of
#the bounding boxes of two entities
#Inputs: ent_a, ent_b - entities
#Return value: real number
def get_centroid_distance(ent_a, ent_b):
    a_centroid = ent_a.bbox_centroid
    b_centroid = ent_b.bbox_centroid
    return point_distance(a_centroid, b_centroid)

#Computes the distance between the centroids of
#the bounding boxes of two entities, normalized by the maximum
#dimension of two entities
#Inputs: ent_a, ent_b - entities
#Return value: real number
def get_centroid_distance_scaled(ent_a, ent_b):
    a_max_dim = max(ent_a.dimensions)
    b_max_dim = max(ent_b.dimensions)

    #add a small number to denominator in order to
    #avoid division by zero in the case when a_max_dim + b_max_dim == 0
    return get_centroid_distance(ent_a, ent_b) / (ent_a.radius + ent_b.radius + 0.0001)


#Computes the distance between two entities in the special
#case if the first entity is elongated, i.e., can be approximated by a line or a rod
#Inputs: ent_a, ent_b - entities
#Return value: real number
def get_line_distance_scaled(ent_a, ent_b):
    a_dims = ent_a.dimensions
    b_dims = ent_b.dimensions
    a_bbox = ent_a.bbox
    dist = 0

    #If ent_a is elongated, one dimension should be much bigger than the sum of the other two
    #Here we check which dimension is that
    if a_dims[0] >= 1.4 * (a_dims[1] + a_dims[2]):
        dist = min(get_distance_from_line(a_bbox[0], a_bbox[4], ent_b.centroid),
                   get_distance_from_line(a_bbox[1], a_bbox[5], ent_b.centroid),
                   get_distance_from_line(a_bbox[2], a_bbox[6], ent_b.centroid),
                   get_distance_from_line(a_bbox[3], a_bbox[7], ent_b.centroid))
        if math.fabs(ent_a.centroid[0] - ent_b.centroid[0]) <= a_dims[0] / 2:
            dist /= ((a_dims[1] + a_dims[2]) / 2 + max(b_dims))
        else:
            dist = math.sqrt(0.5 * (ent_a.centroid[0] - ent_b.centroid[0]) ** 2 + 0.5 * dist ** 2)
    elif a_dims[1] >= 1.4 * (a_dims[0] + a_dims[2]):
        dist = min(get_distance_from_line(a_bbox[0], a_bbox[2], ent_b.centroid),
                   get_distance_from_line(a_bbox[1], a_bbox[3], ent_b.centroid),
                   get_distance_from_line(a_bbox[4], a_bbox[6], ent_b.centroid),
                   get_distance_from_line(a_bbox[5], a_bbox[7], ent_b.centroid))
        if math.fabs(ent_a.centroid[1] - ent_b.centroid[1]) <= a_dims[1] / 2:
            dist /= ((a_dims[0] + a_dims[2]) / 2 + max(b_dims))
        else:
            dist = math.sqrt(0.5 * (ent_a.centroid[1] - ent_b.centroid[1]) ** 2 + 0.5 * dist ** 2)
    elif a_dims[2] >= 1.4 * (a_dims[1] + a_dims[0]):
        dist = min(get_distance_from_line(a_bbox[0], a_bbox[1], ent_b.centroid),
                   get_distance_from_line(a_bbox[2], a_bbox[3], ent_b.centroid),
                   get_distance_from_line(a_bbox[4], a_bbox[5], ent_b.centroid),
                   get_distance_from_line(a_bbox[6], a_bbox[7], ent_b.centroid))
        if math.fabs(ent_a.centroid[2] - ent_b.centroid[2]) <= a_dims[2] / 2:
            dist /= ((a_dims[0] + a_dims[1]) / 2 + max(b_dims))
        else:
            dist = math.sqrt(0.5 * (ent_a.centroid[2] - ent_b.centroid[2]) ** 2 + 0.5 * dist ** 2)
    return dist

#Computes the distance between two entities in the special
#case if the first entity is planar, i.e., can be approximated by a plane or a thin box
#Inputs: ent_a, ent_b - entities
#Return value: real number
def get_planar_distance_scaled(ent_a, ent_b):
    a_dims = ent_a.dimensions
    b_dims = ent_b.dimensions
    a_bbox = ent_a.bbox
    dist = 0

    #If ent_a is planar, one dimension should be much smaller than the other two
    #Here we check which dimension is that
    if a_dims[0] <= 0.5 * a_dims[1] and a_dims[0] <= 0.5 * a_dims[2]:
        dist = min(get_distance_from_plane(ent_b.centroid, a_bbox[0], a_bbox[1], a_bbox[2]),
                   get_distance_from_plane(ent_b.centroid, a_bbox[4], a_bbox[5], a_bbox[6]))
        if math.fabs(ent_a.centroid[1] - ent_b.centroid[1]) <= a_dims[1] / 2 and \
            math.fabs(ent_a.centroid[2] - ent_b.centroid[2]) <= a_dims[2] / 2:
            dist /= (a_dims[0] + max(b_dims))
        else:
            #dist = closest_mesh_distance(ent_a, ent_b)
            dist = math.sqrt(0.6 * ((ent_a.centroid[1] - ent_b.centroid[1]) ** 2 + (ent_a.centroid[2] - ent_b.centroid[2]) ** 2) \
                             + 0.4 * dist ** 2)
    elif a_dims[1] <= 0.5 * a_dims[0] and a_dims[1] <= 0.5 * a_dims[2]:
        dist = min(get_distance_from_plane(ent_b.centroid, a_bbox[0], a_bbox[1], a_bbox[4]),
                   get_distance_from_plane(ent_b.centroid, a_bbox[2], a_bbox[3], a_bbox[5]))
        if math.fabs(ent_a.centroid[0] - ent_b.centroid[0]) <= a_dims[0] / 2 and \
            math.fabs(ent_a.centroid[2] - ent_b.centroid[2]) <= a_dims[2] / 2:
            dist /= (a_dims[1] + max(b_dims))
        else:
            #dist = closest_mesh_distance(ent_a, ent_b)
            dist = math.sqrt(0.6 * ((ent_a.centroid[0] - ent_b.centroid[0]) ** 2 + (ent_a.centroid[2] - ent_b.centroid[2]) ** 2) \
                             + 0.4 * dist ** 2)
    elif a_dims[2] <= 0.5 * a_dims[0] and a_dims[2] <= 0.5 * a_dims[1]:
        dist = min(get_distance_from_plane(ent_b.centroid, a_bbox[0], a_bbox[2], a_bbox[4]),
                   get_distance_from_plane(ent_b.centroid, a_bbox[1], a_bbox[3], a_bbox[5]))
        if math.fabs(ent_a.centroid[1] - ent_b.centroid[1]) <= a_dims[1] / 2 and \
            math.fabs(ent_a.centroid[0] - ent_b.centroid[0]) <= a_dims[0] / 2:
            dist /= (a_dims[2] + max(b_dims))
        else:
            #dist = closest_mesh_distance(ent_a, ent_b)
            dist = math.sqrt(0.6 * ((ent_a.centroid[1] - ent_b.centroid[1]) ** 2 + (ent_a.centroid[0] - ent_b.centroid[0]) ** 2) \
                             + 0.4 * dist ** 2)
    return dist


#Computes the closest distance between the points of two meshes
#Input: ent_a, ent_b - entities
#Return value: real number
def closest_mesh_distance(ent_a, ent_b):
    min_dist = 1e9

    #print (ent_a.vertex_set, ent_b.vertex_set)
    if len(ent_a.vertex_set) * len(ent_b.vertex_set) <= 1000:
        min_dist = min([point_distance(u,v) for u in ent_a.vertex_set for v in ent_b.vertex_set])
        return min_dist
    
    count = 0    
    u0 = ent_a.vertex_set[0]
    v0 = ent_b.vertex_set[0]
    min_dist = point_distance(u0, v0)       
    for v in ent_b.vertex_set:
        if point_distance(u0, v) <= min_dist:
            min_dist = point_distance(u0, v)
            v0 = v
    for u in ent_a.vertex_set:
        if point_distance(u, v0) <= min_dist:
            min_dist = point_distance(u, v0)
            u0 = u
    #lin_dist = min_dist
    #min_dist = 1e9
    #for v in ent_a.total_mesh:
    #    for u in ent_b.total_mesh:
    #        min_dist = min(min_dist, point_distance(u, v))
    #        count += 1
    #print ("COUNT:", count, min_dist, lin_dist)
    return min_dist

#Normalized version of closest_mesh_distance where the distance is scaled
#by the maximum dimensions of two entities
#Input: ent_a, ent_b - entities
#Return value: real number
def closest_mesh_distance_scaled(ent_a, ent_b):
    a_dims = ent_a.dimensions
    b_dims = ent_b.dimensions
    return closest_mesh_distance(ent_a, ent_b) / (max(a_dims) + max(b_dims) + 0.0001)

#Computes the shared volume of the bounding boxes of two entities
#Input: ent_a, ent_b - entities
#Return value: real number
def get_bbox_intersection(ent_a, ent_b):
    span_a = ent_a.span
    span_b = ent_b.span
    int_x = 0
    int_y = 0
    int_z = 0
    if span_a[0] >= span_b[0] and span_a[1] <= span_b[1]:
        int_x = span_a[1] - span_a[0]
    elif span_b[0] >= span_a[0] and span_b[1] <= span_a[1]:
        int_x = span_b[1] - span_b[0]
    elif span_a[0] <= span_b[0] and span_a[1] >= span_b[0] and span_a[1] <= span_b[1]:
        int_x = span_a[1] - span_b[0]
    elif span_a[0] >= span_b[0] and span_a[0] <= span_b[1] and span_a[1] >= span_b[1]:
        int_x = span_b[1] - span_a[0]

    if span_a[2] >= span_b[2] and span_a[3] <= span_b[3]:
        int_y = span_a[3] - span_a[2]
    elif span_b[2] >= span_a[2] and span_b[3] <= span_a[3]:
        int_y = span_b[3] - span_b[2]
    elif span_a[2] <= span_b[2] and span_a[3] >= span_b[2] and span_a[3] <= span_b[3]:
        int_y = span_a[3] - span_b[2]
    elif span_a[2] >= span_b[2] and span_a[2] <= span_b[3] and span_a[3] >= span_b[3]:
        int_y = span_b[3] - span_a[2]

    if span_a[4] >= span_b[4] and span_a[5] <= span_b[5]:
        int_z = span_a[5] - span_a[4]
    elif span_b[4] >= span_a[4] and span_b[5] <= span_a[5]:
        int_z = span_b[5] - span_b[4]
    elif span_a[4] <= span_b[4] and span_a[5] >= span_b[4] and span_a[5] <= span_b[5]:
        int_z = span_a[5] - span_b[4]
    elif span_a[4] >= span_b[4] and span_a[4] <= span_b[5] and span_a[5] >= span_b[5]:
        int_z = span_b[5] - span_a[4]

    vol = int_x * int_y * int_z    
    return vol

#Checks whether the entity is vertically oriented
#Input: ent_a - entity
#Return value: boolean value
def isVertical(ent_a):
    if hasattr(ent_a, 'dimensions'):
        return ent_a.dimensions[0] < 0.5 * ent_a.dimensions[2] or ent_a.dimensions[1] < 0.5 * ent_a.dimensions[2]
    else:
        normal = get_normal(ent_a[0], ent_a[1], ent_a[2])
        return 1 - normal[2]

def cosine_similarity(v1, v2):
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    if l1 == 0 and l2 == 0:
        return 1
    if l1 == 0 or l2 == 0:
        return 0    
    cos = np.dot(v1, v2) / (l1 * l2)
    if cos > 1:
        cos = 1
    if cos < -1:
        cos = -1
    return cos

def within_cone(v1, v2, threshold):
    cos = cosine_similarity(v1, v2)
    #print ("DENOM: {}, TAN: {}".format((1 + np.sign(threshold - cos) * threshold), 0.5 * math.pi * (cos - threshold) / (1 + np.sign(threshold - cos) * threshold)))
    #angle = math.acos(cos)
    tangent = math.tan(0.5 * math.pi * (cos - threshold) / (1 + np.sign(threshold - cos) * threshold))
    if -20 <= tangent and tangent <= 20:
        return 1 / (1 + math.e ** (-tangent))
    elif tangent < -40:
        return 0
    else:
        return 1

def distance(a, b):
    """
    Compute distance between a and b.
    The distance computed depends on the specific object types
    and their geometry.
    """
    
    bbox_a = a.bbox
    bbox_b = b.bbox
    a0 = a.bbox_centroid
    b0 = b.bbox_centroid
    if a.get('extended') is not None:
        return a.get_closest_face_distance(b0)
    if b.get('extended') is not None:
        return b.get_closest_face_distance(a0)

    mesh_dist = closest_mesh_distance_scaled(a, b)
    centroid_dist = get_centroid_distance_scaled(a, b)
    return 0.5 * mesh_dist + 0.5 * centroid_dist

def shared_volume(a, b):
    """Compute shared volume of two entities."""

    #PLACEHOLDER - replace with a more accurate code
    volume = get_bbox_intersection(a, b)

    return volume

def shared_volume_scaled(a, b):
    """Compute shared volume of two entities scaled by their max volume."""

    volume = shared_volume(a, b)
    max_vol = max(a.volume, b.volume)

    if max_vol != 0:
        return volume / max_vol
    elif volume != 0:
        return 1.0
    else:
        return 0.0


def fit_line(points):
    """Compute and return the best-fit line through a set of points."""
    if type(points) == list:
        points = np.array(points)

    if len(points) == 1:
        return points[0], np.array([0, 0, 1.0]), 1

    centroid = np.mean(points, axis=0)
    
    #Translating the points to the origin
    X = points - centroid

    Sigma = np.cov(X.T)
    U, S, V = np.linalg.svd(Sigma)
    D = U[:,0]

    #Project the points along the largest eigenvector
    proj = D.T.dot(X.T)

    #Get 3-D coordinates of the projected points
    X_proj = np.outer(D, proj.T).T
    
    #print (X, "\n", X_proj)

    #Deviations of the original points from the projected points
    dev = [np.linalg.norm(v) for v in X - X_proj]
    
    avg_dist = np.average(dev)
    max_dist = np.max(dev)

    #Normalize the direction vector
    D = D / np.linalg.norm(D)
    
    #Reorient left-to-right and down-to-up if direction is closely aligned with z or x axis
    if D.dot(np.array([0, 0, 1.0])) > 0.71 or D.dot(np.array([1.0, 0, 0])) < -0.71:
        D = -D
        
    return centroid, D, avg_dist, max_dist


def rotation_matrix(alpha, beta, gamma):
    """
    Compute the rotation matrix.

    params:
    alpha, beta, gamma - rotation angles around the x, y and z axes,
    respecticaly. The angles are in radians.

    return: the 3x3 rotation matrix
    """

    #Unit test: rotation_matrix(0, -math.pi/4, -math.pi/4).dot(np.array([1,0,0])) = [0.5, 0.5, 0.7071]
    
    cosa = math.cos(alpha)
    cosb = math.cos(beta)
    cosg = math.cos(gamma)
    sina = math.sin(alpha)
    sinb = math.sin(beta)
    sing = math.sin(gamma)
    R = np.array([[cosb*cosg, cosa*sing - sina*cosg*sinb, sina*sing + cosa*cosg*sinb],\
                     [-sing*cosb, cosg*cosa + sing*sinb*sina, cosg*sina - sing*sinb*cosa],\
                     [-sinb, -sina*cosb, cosa*cosb]])

    return R

def get_axis_angles(vect):
    vx = vect[0]
    vy = vect[1]
    vz = vect[2]

    if vz == 0 and vy == 0:
        return (math.pi / 2, 0, math.pi / 2)

    if vz == 0 and vx == 0:
        return (math.pi / 2, 0, 0)

    if vx == 0 and vy == 0:
        return (0, 0, 0)

    cosa = vz / math.sqrt(vz*vz + vy*vy)
    alpha = math.acos(cosa)
    if vy > 0:
        alpha = -alpha

    cosb = vz / math.sqrt(vz*vz + vx*vx)
    beta = math.acos(cosb)
    if vx > 0:
        beta = -beta

    cosg = vy / math.sqrt(vy*vy + vx*vx)
    gamma = math.acos(cosg)
    if vx > 0:
        gamma = -gamma

    return alpha, beta, gamma

def eye_projection(point, up, right, focus_dist, eye_dist):
    """
    Compute the projection of an object onto the observer's visual plane.

    """
    scaling_factor = (focus_dist - eye_dist) / focus_dist
    up0 = scaling_factor * point.dot(up) / np.linalg.norm(up)
    right0 = scaling_factor * point.dot(right) / np.linalg.norm(right)
    return right0, up0

def signed_point_to_plane_dist(point, face):
    unit_norm = get_normal(face[0], face[1], face[2])
    dist = np.dot(unit_norm, point - face[0])
    print ("SIGNED PLANE DISTANCE: ", dist)
    return dist

def unsigned_point_to_plane_dist(point, face):
    dist = math.fabs(signed_point_to_face_dist(point, face))
    print ("UNSIGNED PLANE DISTANCE: ", dist)
    return dist

def is_in_face(point, face):
    if type(point) != np.ndarray:
        point = np.array(point)
    queue = [np.array(item) for item in face]
    queue.append(queue[0])
    angle = 0
    for i in range(len(queue) - 1):
        v1 = queue[i] - point
        v2 = queue[i+1] - point
        if np.linalg.norm(v1) < 0.00001 or np.linalg.norm(v2) < 0.00001:
            return 1
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)        
        cosa = np.dot(v1, v2)
        if cosa < -1:
            cosa = -1
        elif cosa > 1:
            cosa = 1
        angle += math.acos(cosa)
#    print (angle)
    return math.e ** (- math.fabs(angle - 2 * math.pi))

def camera_matrix(location, direction):
    pass

def projection_bbox_center(bbox):
    return np.array([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2])

def projection_bbox_area(bbox):
    return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])

def get_2d_size(bbox):
    center = projection_bbox_center(bbox)
    x_max = bbox[1]
    x_min = bbox[0]
    y_max = bbox[3]
    y_min = bbox[2]
    diag = np.linalg.norm(np.array([x_max - x_min, y_max - y_min]))
    radius = diag / 2
    return radius


import bpy
import bmesh

def bmesh_copy_from_object(obj, transform=True, triangulate=True, apply_modifiers=False):
    """
    Returns a transformed, triangulated copy of the mesh
    """

    assert(obj.type == 'MESH')

    if apply_modifiers and obj.modifiers:
        me = obj.to_mesh(bpy.context.scene, True, 'PREVIEW', calc_tessface=False)
        bm = bmesh.new()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    else:
        me = obj.data
        if obj.mode == 'EDIT':
            bm_orig = bmesh.from_edit_mesh(me)
            bm = bm_orig.copy()
        else:
            bm = bmesh.new()
            bm.from_mesh(me)

    # Remove custom data layers to save memory
    for elem in (bm.faces, bm.edges, bm.verts, bm.loops):
        for layers_name in dir(elem.layers):
            if not layers_name.startswith("_"):
                layers = getattr(elem.layers, layers_name)
                for layer_name, layer in layers.items():
                    layers.remove(layer)

    if transform:
        bm.transform(obj.matrix_world)

    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)

    return bm

def bmesh_check_intersect_objects(obj, obj2):
    """
    Check if any faces intersect with the other object

    returns a boolean
    """
    assert(obj != obj2)

    # Triangulate
    bm = bmesh_copy_from_object(obj, transform=True, triangulate=True)
    bm2 = bmesh_copy_from_object(obj2, transform=True, triangulate=True)

    # If bm has more edges, use bm2 instead for looping over its edges
    # (so we cast less rays from the simpler object to the more complex object)
    if len(bm.edges) > len(bm2.edges):
        bm2, bm = bm, bm2

    # Create a real mesh (lame!)
    scene = bpy.context.scene
    me_tmp = bpy.data.meshes.new(name="~temp~")
    bm2.to_mesh(me_tmp)
    bm2.free()
    obj_tmp = bpy.data.objects.new(name=me_tmp.name, object_data=me_tmp)
    scene.objects.link(obj_tmp)
    scene.update()
    ray_cast = obj_tmp.ray_cast

    intersect = False

    EPS_NORMAL = 0.000001
    EPS_CENTER = 0.01  # should always be bigger

    #for ed in me_tmp.edges:
    for ed in bm.edges:
        v1, v2 = ed.verts

        # setup the edge with an offset
        co_1 = v1.co.copy()
        co_2 = v2.co.copy()
        co_mid = (co_1 + co_2) * 0.5
        no_mid = (v1.normal + v2.normal).normalized() * EPS_NORMAL
        co_1 = co_1.lerp(co_mid, EPS_CENTER) + no_mid
        co_2 = co_2.lerp(co_mid, EPS_CENTER) + no_mid

        co, no, index = ray_cast(co_1, co_2)
        if index != -1:
            intersect = True
            break

    scene.objects.unlink(obj_tmp)
    bpy.data.objects.remove(obj_tmp)
    bpy.data.meshes.remove(me_tmp)

    scene.update()
    return intersect


#obj = bpy.context.object
#obj2 = (ob for ob in bpy.context.selected_objects if ob != obj).__next__()
#intersect = bmesh_check_intersect_objects(obj, obj2)

#print("There are%s intersections." % ("" if intersect else " NO"))

def intersection_check(a, b):
    """
    Check if two Blender objects are intersecting.

    Input: 
    a,b - blender objects

    Return: 
    True if the meshes of a and b intersect each other, False otherwise

    """

    #Create and fill bmesh data
    
    bm1 = bmesh.new()
    bm1.from_mesh(a.data)
    bm2 = bmesh.new()
    bm2.from_mesh(b.data)            

    bm1.transform(a.matrix_world)
    bm2.transform(b.matrix_world) 

    #Create BVH trees based on bmeshes
    a_BVHtree = BVHTree.FromBMesh(bm1)
    b_BVHtree = BVHTree.FromBMesh(bm2)           

    #Check if the overlap set is empty and return the result
    return a_BVHtree.overlap(b_BVHtree) != []

def intersection_entities(a, b):
    for acomp in a.components:
        for bcomp in b.components:
            if intersection_check(acomp, bcomp):
                return True
            
    return False
