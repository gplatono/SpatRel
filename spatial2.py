import numpy as np
import math
#import spatial
#from constraint_solver import *
from entity import Entity
from geometry_utils import *
from queue import Queue
from mathutils import Vector
# import bpy_extras
from functools import reduce
import itertools

observer = None

class Spatial:
    def __init__(self):
        self.pred_to_str = {
            spatial.on: 'on',
            spatial.to_the_left_of_deic: 'to the left of',
            spatial.to_the_right_of_deic: 'to the right of',
            spatial.near: 'near',
            spatial.above: 'above',
            spatial.below: 'below',
            spatial.over: 'over',
            spatial.inside: 'in',
            spatial.touching: 'touching',
            spatial.at: 'next to',
            spatial.higher_than: 'high',
            spatial.lower_than: 'low',
            spatial.in_front_of_deic: 'in front of',
            spatial.behind: 'behind',
            spatial.clear: 'clear',
            spatial.where: 'where',
            exist: 'exist',
            spatial.facing: 'facing',
            color_pred: 'color',
            blue: 'blue',
            red: 'red',
            green: 'green', }
        self.str_to_pred = {
            'on.p': spatial.on,
            'on': spatial.on,

            'to_the_left_of.p': spatial.to_the_left_of_deic,
            'left.a': spatial.to_the_left_of_deic,
            'leftmost.a': spatial.to_the_left_of_deic,
            'to_the_right_of.p': spatial.to_the_right_of_deic,
            'right.a': spatial.to_the_right_of_deic,
            'rightmost.a': spatial.to_the_right_of_deic,
            'right.p': spatial.to_the_right_of_deic,
            'left.p': spatial.to_the_left_of_deic,

            'near.p': spatial.near,
            'near_to.p': spatial.near,
            'close_to.p': spatial.near,
            'close.a': spatial.near,
            'on.p': spatial.on,
            'on_top_of.p': spatial.on,
            'above.p': spatial.above,
            'below.p': spatial.below,
            'over.p': spatial.over,
            'under.p': spatial.below,
            'underneath.p': spatial.below,
            'supporting.p': spatial.under,

            'in.p': spatial.inside,
            'in': spatial.inside,
            'inside.p': spatial.inside,

            'touching.p': spatial.touching,
            'touch.v': spatial.touching,
            'adjacent_to.p': spatial.touching,

            'at.p': spatial.at,
            'next_to.p': spatial.at,

            'high.a': spatial.higher_than,
            'upper.a': spatial.higher_than,
            'highest.a': spatial.higher_than,
            'topmost.a': spatial.higher_than,
            'top.a': spatial.higher_than,
            'low.a': spatial.lower_than,
            'lowest.a': spatial.lower_than,

            'in_front_of.p': spatial.in_front_of,
            'front.a': spatial.in_front_of,
            'frontmost.a': spatial.in_front_of,

            'behind.p': spatial.behind,
            'backmost.a': spatial.behind,
            'back.a': spatial.behind,
            'farthest.a': spatial.behind,
            'far.a': spatial.behind,
            'between.p': spatial.between,
            'clear.a': spatial.clear,
            'where.a': spatial.where,
            'exist.pred': exist,

            'face.v': spatial.facing,
            'facing.p': spatial.facing,

            'color.pred': color_pred,

            'blue.a': blue,
            'red.a': red,
            'green.a': green,

            'blue': blue,
            'red': red,
            'green': green, }
        self.arity = {
            spatial.on: 2,
            spatial.to_the_left_of_deic: 2,
            spatial.to_the_right_of_deic: 2,
            spatial.near: 2,
            spatial.above: 2,
            spatial.below: 2,
            spatial.over: 2,
            spatial.under: 2,
            spatial.inside: 2,
            spatial.touching: 2,
            spatial.at: 2,
            spatial.in_front_of: 2,
            spatial.behind: 2,
            spatial.between: 3,
            spatial.clear: 1,
            spatial.higher_than: 2,
            spatial.lower_than: 2,
            spatial.where: 1,
            spatial.facing: 2,
            ident: 2,
            exist: 1,
            color_pred: 1,

            blue: 1,
            red: 1,
            green: 1, }

class Node:
    def __init__(self):
        self.arity = 2
        self.connections = []

    def set_connections(self, connections):
        self.connections = connections

    def get_connections(self):
        return self.connections

class ConeArea(Node):
    """
	Factor determining containment in an infinite cone-shaped beam
	with the given width, emanating from the origin object/point into a given
	direction.
	"""

    def __init__(self, origin, direction, width):
        self.direction = direction
        self.width = width

    def compute(self, vect):
        cos = self.direction.dot(vect) / (np.linalg.norm(self.direction) * \
                                          np.linalg.norm(vect))
        angle = math.acos(cos)
        final_score = 1 / (1 + math.e ** (2 * (self.width - angle)))
        return final_score


class FrameSize(Node):
    """
	Factor representing the frame size of the current scene.

	"""

    def compute(self):
        max_x = -100
        min_x = 100
        max_y = -100
        min_y = 100
        max_z = -100
        min_z = 100

        # Computes the scene bounding box
        for entity in entities:
            max_x = max(max_x, entity.span[1])
            min_x = min(min_x, entity.span[0])
            max_y = max(max_y, entity.span[3])
            min_y = min(min_y, entity.span[2])
            max_z = max(max_z, entity.span[5])
            min_z = min(min_z, entity.span[4])
        return max(max_x - min_x, max_y - min_y, max_z - min_z)


class RawDistance(Node):

    def compute(self, tr, lm):
        dist = dist_obj(tr, lm)
        if tr.get('planar') is not None:
            # print ("TEST", a.name, b.name)
            dist = min(dist, get_planar_distance_scaled(tr, lm))
        elif lm.get('planar') is not None:
            dist = min(dist, get_planar_distance_scaled(lm, tr))
        elif tr.get('vertical_rod') is not None or tr.get('horizontal_rod') is not None or tr.get('rod') is not None:
            dist = min(dist, get_line_distance_scaled(tr, lm))
        elif lm.get('vertical_rod') is not None or lm.get('horizontal_rod') is not None or lm.get('rod') is not None:
            dist = min(dist, get_line_distance_scaled(lm, tr))
        elif tr.get('concave') is not None or lm.get('concave') is not None:
            dist = min(dist, closest_mesh_distance_scaled(tr, lm))

        final_score = dist
        return final_score


class RightOf_Deictic(Node):
    """Deictic sense of the to-the-right-of relation."""

    def compute(self, tr, lm):
        tr_bbox = get_2d_bbox(vp_project(tr, observer))
        lm_bbox = get_2d_bbox(vp_project(lm, observer))
        axial_dist = scaled_axial_distance(tr_bbox, lm_bbox)
        if axial_dist[0] <= 0:
            return 0

        horizontal_component = asym_inv_exp(axial_dist[0], 1, 1, 0.05)
        vertical_component = math.exp(- math.fabs(axial_dist[1]))

        final_score = 0.9 * (0.4 * horizontal_component + 0.6 * vertical_component)
        return final_score


class RightOf_Extrinsic(Node):
    """Extrinsic sense of the to-the-right-of relation."""

    def compute(self, tr, lm):
        disp_vec = np.array(tr.bbox_centroid - lm.bbox_centroid)
        dist = np.linalg.norm(disp_vec)
        disp_vec = disp_vec / dist

        extrinsic_right = np.array([1, 0, 0])
        cos = extrinsic_right.dot(disp_vec)

        final_score = math.e ** (- 0.1 * (1 - cos)) * math.e ** (- 0.05 * dist / max(tr.size, lm.size))
        return final_score


class RightOf_Intrinsic(Node):
    """Intrinsic sense of the to-the-right-of relation."""

    def compute(self, tr, lm):
        disp_vec = np.array(tr.bbox_centroid - lm.bbox_centroid)
        dist = np.linalg.norm(disp_vec)
        disp_vec = disp_vec / dist

        intrinsic_right = np.array(lm.right)
        cos = intrinsic_right.dot(disp_vec)

        final_score = math.e ** (- 0.1 * (1 - cos)) * math.e ** (- 0.05 * dist / max(tr.size, lm.size))
        return final_score


class RightOf(Node):
    """Implementation of the general to-the-right-of relation."""

    def __init__(self):
        deictic_measure = RightOf_Deictic()
        extrinsic_measure = RightOf_Extrinsic()
        intrinsic_measure = RightOf_Intrinsic()
        self.set_connections([deictic_measure, extrinsic_measure, intrinsic_measure])
    
    def compute(self, tr, lm=None):
        ret_val = 0
        if type(tr) == Entity and type(lm) == Entity:
            connections = self.get_connections()
            return max(connections[0].compute(tr, lm), connections[1].compute(tr, lm),
                       connections[2].compute(tr, lm))
        elif lm is None:
            ret_val = np.avg([self.compute(tr, entity) for entity in world.active_context])
        return ret_val


class LeftOf_Deictic(Node):
    def __init__(self):
        self.set_connections([RightOf_Deictic()])

    def compute(self, tr, lm):
        return self.get_connections()[0].compute(tr=lm, lm=tr)


class LeftOf_Extrinsic(Node):
    def __init__(self):
        self.set_connections([RightOf_Extrinsic()])

    def compute(self, tr, lm):
        return self.get_connections()[0].compute(tr=lm, lm=tr)

class LeftOf_Intrinsic(Node):
    def __init__(self):
        self.set_connections([RightOf_Intrinsic()])

    def compute(self, tr, lm):
        return self.get_connections()[0].compute(tr=lm, lm=tr)


class LeftOf(Node):
    def __init__(self):
        deictic_measure = LeftOf_Deictic()
        extrinsic_measure = LeftOf_Extrinsic()
        intrinsic_measure = LeftOf_Intrinsic()
        self.set_connections([deictic_measure, extrinsic_measure, intrinsic_measure])       

    def compute(self, tr, lm=None):
        ret_val = 0
        if type(tr) == Entity and type(lm) == Entity:
            connections = self.get_connections()
            return max(connections[0].compute(tr, lm), connections[1].compute(tr, lm),
                       connections[2].compute(tr, lm))
        elif lm is None:
            ret_val = np.avg([self.compute(tr, entity) for entity in world.active_context])
        return ret_val


class InFrontOf_Deictic(Node):
    def compute(self, tr, lm):
        # def in_front_of_extr(a, b, observer):
        bbox_tr = tr.bbox
        max_dim_a = max(bbox_tr[7][0] - bbox_tr[0][0],
                        bbox_tr[7][1] - bbox_tr[0][1],
                        bbox_tr[7][2] - bbox_tr[0][2]) + 0.0001
        dist = get_distance_from_line(world.get_observer().centroid, lm.centroid, tr.centroid)
        # print ("{}, {}, CLOSER: {}, WC_DEIC: {}, WC_EXTR: {}, DIST: {}".format(a.name, b.name, closer_than(a, b, observer), within_cone(b.centroid - observer.centroid, a.centroid - observer.centroid, 0.95), within_cone(b.centroid - a.centroid, Vector((0, -1, 0)) - a.centroid, 0.8), e ** (- 0.1 * get_centroid_distance_scaled(a, b))))
        # print ("WITHIN CONE:")
        a_bbox = get_2d_bbox(vp_project(tr, world.get_observer()))
        b_bbox = get_2d_bbox(vp_project(lm, world.get_observer()))
        a_center = projection_bbox_center(a_bbox)
        b_center = projection_bbox_center(b_bbox)
        dist = np.linalg.norm(a_center - b_center)
        scaled_proj_dist = dist / (max(get_2d_size(a_bbox), get_2d_size(b_bbox)) + 0.001)
        # scaled_proj_dist = gaussian(scaled_proj_dist, 0, 1)

        # print ("BBOX :", a_bbox, b_bbox)
        # print ("PROJ DIST:" ,scaled_proj_dist)
        a_dist = np.linalg.norm(tr.location - world.observer.location)
        b_dist = np.linalg.norm(lm.location - world.observer.location)
        # print ("SIGM, OVERLAP :  ", sigmoid(b_dist - a_dist, 1, 0.5), math.e ** (-0.5 * scaled_proj_dist))
        return 0.5 * (sigmoid(b_dist - a_dist, 1, 0.5) + math.e ** (-0.5 * scaled_proj_dist))
        # return closer_than(a, b, world.observer) * math.e ** (-0.1 * scaled_dist)
        # return math.e ** (- 0.01 * get_centroid_distance_scaled(a, b)) * within_cone(b.centroid - a.centroid, world.front_axis, 0.7)
        # return math.e ** (- 0.01 * get_centroid_distance_scaled(a, b)) * within_cone(b.centroid - a.centroid, Vector((1, 0, 0)), 0.7)
        '''0.3 * closer_than(a, b, observer) + \
                      0.7 * (max(within_cone(b.centroid - observer.centroid, a.centroid - observer.centroid, 0.95),
                      within_cone(b.centroid - a.centroid, Vector((1, 0, 0)), 0.7)) * \
                      e ** (- 0.2 * get_centroid_distance_scaled(a, b)))#e ** (-dist / max_dim_a))'''


class InFrontOf_Extrinsic(Node):
    def compute(self, tr, lm):
        # proj_dist = math.fabs(world.front_axis.dot(a.location)) - math.fabs(world.front_axis.dot(b.location))
        # proj_dist_scaled = proj_dist / max(a.size, b.size)
        # print ("PROJ_DISTANCE", proj_dist_scaled)
        return math.e ** (- 0.01 * get_centroid_distance_scaled(tr, lm)) * within_cone(lm.centroid - tr.centroid,
                                                                                 -world.front_axis, 0.7)


class InFrontOf_Intrinsic(Node):
    def compute(self, tr, lm):
        return math.e ** (- 0.01 * get_centroid_distance_scaled(tr, lm)) * within_cone(lm.centroid - tr.centroid,
                                                                                       -lm.front, 0.7)


class InFrontOf(Node):
    def __init__(self):
        deictic_measure = InFrontOf_Deictic()
        extrinsic_measure = InFrontOf_Extrinsic()
        intrinsic_measure = InFrontOf_Intrinsic()
        self.set_connections([deictic_measure, extrinsic_measure, intrinsic_measure])

    def compute(self, tr, lm=None):
        ret_val = 0
        if type(tr) == Entity and type(lm) == Entity:
            connections = self.get_connections()
            return max(connections[0].compute(tr, lm), connections[1].compute(tr, lm),
                       connections[2].compute(tr, lm))
        elif lm is None:
            ret_val = np.avg([self.compute(tr, entity) for entity in world.active_context])
        return ret_val


class Behind_Deictic(Node):
    def __init__(self):
        self.set_connections([InFrontOf_Deictic()])

    def compute(self, tr, lm):
        return self.get_connections()[0].compute(tr=lm, lm=tr)


class Behind_Extrinsic(Node):
    def __init__(self):
        self.set_connections([InFrontOf_Extrinsic()])

    def compute(self, tr, lm):
        return self.get_connections()[0].compute(tr=lm, lm=tr)


class Behind_Intrinsic(Node):
    def __init__(self):
        self.set_connections([InFrontOf_Intrinsic()])

    def compute(self, tr, lm):
        return self.get_connections()[0].compute(tr=lm, lm=tr)


class Behind(Node):
    def __init__(self):
        deictic_measure = Behind_Deictic()
        extrinsic_measure = Behind_Extrinsic()
        intrinsic_measure = Behind_Intrinsic()
        self.set_connections([deictic_measure, extrinsic_measure, intrinsic_measure])

    def compute(self, tr, lm=None):
        ret_val = 0
        if type(tr) == Entity and type(lm) == Entity:
            connections = self.get_connections()
            return max(connections[0].compute(tr, lm), connections[1].compute(tr, lm),
                       connections[2].compute(tr, lm))
        elif lm is None:
            ret_val = np.avg([self.compute(tr, entity) for entity in world.active_context])
        return ret_val

# =======================================================================================================
# ====================================OLD CODE STARTS HERE===============================================
# =======================================================================================================
entities = []
world = None


def dist_obj(a, b):
    if type(a) is not Entity or type(b) is not Entity:
        return -1
    bbox_a = a.bbox
    bbox_b = b.bbox
    center_a = a.bbox_centroid
    center_b = b.bbox_centroid
    if a.get('extended') is not None:
        return a.get_closest_face_distance(center_b)
    if b.get('extended') is not None:
        return b.get_closest_face_distance(center_a)
    return point_distance(center_a, center_b)


# Computes the normalized area of the intersection of projection of two entities onto the XY-plane
# Inputs: a, b - entities
# Return value: real number
def get_proj_intersection(a, b):
    bbox_a = a.bbox
    bbox_b = b.bbox
    axmin = a.span[0]
    axmax = a.span[1]
    aymin = a.span[2]
    aymax = a.span[3]
    bxmin = b.span[0]
    bxmax = b.span[1]
    bymin = b.span[2]
    bymax = b.span[3]
    xdim = 0
    ydim = 0
    if axmin >= bxmin and axmax <= bxmax:
        xdim = axmax - axmin
    elif bxmin >= axmin and bxmax <= axmax:
        xdim = bxmax - bxmin
    elif axmin <= bxmin and axmax <= bxmax and axmax >= bxmin:
        xdim = axmax - bxmin
    elif axmin >= bxmin and axmin <= bxmax and axmax >= bxmax:
        xdim = bxmax - axmin

    if aymin >= bymin and aymax <= bymax:
        ydim = aymax - aymin
    elif bymin >= aymin and bymax <= aymax:
        ydim = bymax - bymin
    elif aymin <= bymin and aymax <= bymax and aymax >= bymin:
        ydim = aymax - bymin
    elif aymin >= bymin and aymin <= bymax and aymax >= bymax:
        ydim = bymax - aymin
    area = xdim * ydim

    # Normalize the intersection area to [0, 1]
    return math.e ** (area - min((axmax - axmin) * (aymax - aymin), (bxmax - bxmin) * (bymax - bymin)))


# Returns the orientation of the entity relative to the coordinate axes
# Inputs: a - entity
# Return value: triple representing the coordinates of the orientation vector
def get_planar_orientation(a):
    dims = a.dimensions
    if dims[0] == min(dims):
        return (1, 0, 0)
    elif dims[1] == min(dims):
        return (0, 1, 0)
    else:
        return (0, 0, 1)


# Computes the degree of vertical alignment (coaxiality) between two entities
# The vertical alignment takes the max value if one of the objects is directly above the other
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def v_align(a, b):
    dim_a = a.dimensions
    dim_b = b.dimensions
    center_a = a.bbox_centroid
    center_b = b.bbox_centroid
    return gaussian(0.9 * point_distance((center_a[0], center_a[1], 0), (center_b[0], center_b[1], 0)) /
                    (max(dim_a[0], dim_a[1]) + max(dim_b[0], dim_b[1])), 0, 1 / math.sqrt(2 * math.pi))


# Computes the degree of vertical offset between two entities
# The vertical offset measures how far apart are two entities one
# of which is above the other. Takes the maximum value when one is
# directly on top of another
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def v_offset(a, b):
    dim_a = a.dimensions
    dim_b = b.dimensions
    center_a = a.bbox_centroid
    center_b = b.bbox_centroid
    h_dist = math.sqrt((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2)
    return gaussian(2 * (center_a[2] - center_b[2] - 0.5 * (dim_a[2] + dim_b[2])) / \
                    (1e-6 + dim_a[2] + dim_b[2]), 0, 1 / math.sqrt(2 * math.pi))


def scaled_axial_distance(a_bbox, b_bbox):
    a_span = (a_bbox[1] - a_bbox[0], a_bbox[3] - a_bbox[2])
    b_span = (b_bbox[1] - b_bbox[0], b_bbox[3] - b_bbox[2])
    a_center = ((a_bbox[0] + a_bbox[1]) / 2, (a_bbox[2] + a_bbox[3]) / 2)
    b_center = ((b_bbox[0] + b_bbox[1]) / 2, (b_bbox[2] + b_bbox[3]) / 2)
    axis_dist = (a_center[0] - b_center[0], a_center[1] - b_center[1])
    # print ("SPANS:", a_span, b_span, a_center, b_center)
    return (axis_dist[0] / (max(a_span[0], b_span[0]) + 0.01), axis_dist[1] / (max(a_span[1], b_span[1]) + 0.01))


# Computes the projection of an entity onto the observer's visual plane
# Inputs: entity - entity, observer - object, representing observer's position
# and orientation
# Return value: list of pixel coordinates in the observer's plane if vision
def vp_project(entity, observer):
    # points = reduce((lambda x,y: x + y), [[obj.matrix_world * v.co for v in obj.data.vertices] for obj in entity.constituents if (obj is not None and hasattr(obj.data, 'vertices') and hasattr(obj, 'matrix_world'))])
    # co_2d = [bpy_extras.object_utils.world_to_camera_view(world.scene, observer.camera, point) for point in points]
    # render_scale = world.scene.render.resolution_percentage / 100
    # render_size = (int(world.scene.render.resolution_x * render_scale), int(world.scene.render.resolution_y * render_scale),)
    # pixel_coords = [(round(point.x * render_size[0]),round(point.y * render_size[1]),) for point in co_2d]
    pixel_coords = [(eye_projection(point, observer.up, observer.right, np.linalg.norm(observer.location), 2)) for point
                    in entity.vertex_set]
    return pixel_coords


# ==========================================================================================
# Raw metric for the nearness relation
# Doesn't take into account the nearness statistics in the scene
# Inputs: a, b - entities
# Return value: real number from [0, 1], the raw nearness measure
def near_raw(a, b):
    bbox_a = a.bbox
    bbox_b = b.bbox
    dist = dist_obj(a, b)
    max_dim_a = max(bbox_a[7][0] - bbox_a[0][0],
                    bbox_a[7][1] - bbox_a[0][1],
                    bbox_a[7][2] - bbox_a[0][2])
    max_dim_b = max(bbox_b[7][0] - bbox_b[0][0],
                    bbox_b[7][1] - bbox_b[0][1],
                    bbox_b[7][2] - bbox_b[0][2])
    if a.get('planar') is not None:
        # print ("TEST", a.name, b.name)
        dist = min(dist, get_planar_distance_scaled(a, b))
    elif b.get('planar') is not None:
        dist = min(dist, get_planar_distance_scaled(b, a))
    elif a.get('vertical_rod') is not None or a.get('horizontal_rod') is not None or a.get('rod') is not None:
        dist = min(dist, get_line_distance_scaled(a, b))
    elif b.get('vertical_rod') is not None or b.get('horizontal_rod') is not None or b.get('rod') is not None:
        dist = min(dist, get_line_distance_scaled(b, a))
    elif a.get('concave') is not None or b.get('concave') is not None:
        dist = min(dist, closest_mesh_distance_scaled(a, b))

    fr_size = spatial.get_frame_size(entities)
    raw_metric = math.e ** (- 0.1 * dist)
    '''0.5 * (1 - min(1, dist / avg_dist + 0.01) +'''
    print("RAW NEAR: ", a, b, raw_metric * (1 - raw_metric / fr_size))
    return raw_metric * (1 - raw_metric / fr_size)


entities = None


# Computes the nearness measure for two entities
# Takes into account the scene statistics:
# The raw nearness score is updated depending on whether one object is the closest to another
# Inputs: a, b - entities
# Return value: real number from [0, 1], the nearness measure
def near(a, b):
    # entities = get_entities()
    # print (entities)
    if a == b:
        return 0
    raw_near_a = []
    raw_near_b = []
    raw_near_measure = near_raw(a, b)
    for entity in entities:
        if entity != a and entity != b:
            near_a_entity = near_raw(a, entity)
            near_b_entity = near_raw(b, entity)
            # print (entity.name, near_a_entity, near_b_entity)
            # if dist_a_to_entity < raw_dist:
            raw_near_a += [near_a_entity]
            # if dist_b_to_entity < raw_dist:
            raw_near_b += [near_b_entity]
    # print ("RAW_NEAR_A: ", raw_near_a, entities)
    # print ("RAW:", a.name, b.name, raw_near_measure)
    average_near_a = sum(raw_near_a) / len(raw_near_a)
    average_near_b = sum(raw_near_b) / len(raw_near_b)
    avg_near = 0.5 * (average_near_a + average_near_b)
    max_near_a = max(raw_near_a)
    max_near_b = max(raw_near_b)
    max_near = max(raw_near_measure, max_near_a, max_near_b)
    # print ("AVER: ", average_near_a, average_near_b)
    ratio = raw_near_measure / max_near
    if (raw_near_measure < avg_near):
        near_measure_final = 0.5 * raw_near_measure
    else:
        near_measure_final = raw_near_measure * ratio
    near_measure = raw_near_measure + (raw_near_measure - avg_near) * min(raw_near_measure, 1 - raw_near_measure)
    # print ("RAW: {}; NEAR: {}; FINAL: {}; AVER: {};".format(raw_near_measure, near_measure, near_measure_final, (average_near_a + average_near_b) / 2))
    return near_measure


# Computes the between relation (a is between b and c)
# Inputs: a, b, c - entities
# Return value: real number from [0, 1]
def between(a, b, c):
    print("ENTERING THE BETWEEN...", a, b, c)
    bbox_a = a.bbox
    bbox_b = a.bbox
    bbox_c = c.bbox
    center_a = a.bbox_centroid
    center_b = b.bbox_centroid
    center_c = c.bbox_centroid
    # print ("1")
    vec1 = np.array(center_b) - np.array(center_a)
    vec2 = np.array(center_c) - np.array(center_a)
    # print ("2", )
    # print (np.dot(vec1, vec2))
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 0.001)
    # print (cos, max([max(a.dimensions), max(b.dimensions), max(c.dimensions)]))
    # dist = get_distance_from_line(center_b, center_c, center_a) / max([max(a.dimensions), max(b.dimensions), max(c.dimensions)])
    scaled_dist = np.linalg.norm(b.bbox_centroid - c.bbox_centroid) / (2 * a.size)
    dist_coeff = math.exp(-0.05 * scaled_dist)
    # print ("3")
    # print ("\nFINAL VALUE BETWEEN: ", a , b, c, math.exp(- math.fabs(-1 - cos)))
    print("BETWEEN DIST FACT: ", dist_coeff)
    ret_val = math.exp(- math.fabs(-1 - cos)) * dist_coeff
    return ret_val


# Computes the "larger-than" relation
# Inputs: a, b - entities
# Return value: real number from [0, 0.5]
def larger_than(a, b):
    bbox_a = a.bbox
    bbox_b = b.bbox
    return 1 / (1 + math.e ** (bbox_b[7][0] - bbox_b[0][0] \
                               + bbox_b[7][1] - bbox_b[0][1] \
                               + bbox_b[7][2] - bbox_b[0][2] \
                               - (bbox_a[7][0] - bbox_a[0][0] \
                                  + bbox_a[7][1] - bbox_a[0][1] \
                                  + bbox_a[7][2] - bbox_a[0][2])))


# Computes the "on" relation
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def on(a, b):
    if a == b:
        return 0
    proj_dist = np.linalg.norm(np.array([a.location[0] - b.location[0], a.location[1] - b.location[1]]))
    proj_dist_scaled = proj_dist / (max(a.size, b.size) + 0.01)
    print("LOCA: ", proj_dist_scaled)
    hor_offset = math.e ** (-0.3 * proj_dist_scaled)
    # print ("PROJ DIST: ", a, b, hor_offset)

    ret_val = touching(a, b) * above(a, b) * hor_offset if hor_offset < 0.9 else above(a, b)  # * touching(a, b)

    # print ("CURRENT ON: ", a, b, ret_val, above(a, b), touching(a, b), hor_offset)
    #    ret_val =  touching(a, b) * hor_offset if above(a, b) < 0.88 else above(a, b) * touching(a, b)
    # print ("CURRENT ON:", ret_val)
    if b.get('planar') is not None and larger_than(b, a) and a.centroid[2] > 0.5 * a.dimensions[2]:
        ret_val = max(ret_val, touching(a, b))
    # ret_val = 0.5 * (v_offset(a, b) + get_proj_intersection(a, b))
    # print ("ON {}, {}, {}".format(ret_val, get_proj_intersection(a, b), v_offset(a, b)))
    # ret_val = max(ret_val, 0.5 * (above(a, b) + touching(a, b)))
    # print ("ON {}".format(ret_val))
    for ob in b.constituents:
        ob_ent = Entity(ob)
        if ob.get('working_surface') is not None or ob.get('planar') is not None:
            ret_val = max(ret_val, 0.5 * (v_offset(a, ob_ent) + get_proj_intersection(a, ob_ent)))
            ret_val = max(ret_val, 0.5 * (int(near(a, ob_ent) > 0.99) + larger_than(ob_ent, a)))
    if b.get('planar') is not None and isVertical(b):
        ret_val = max(ret_val, math.exp(- 0.5 * get_planar_distance_scaled(a, b)))
    return ret_val


# Computes the "over" relation
# Currently, the motivation behind the model is that
# one object is considered to be over the other
# iff it's above it and relatively close to it.
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def over(a, b):
    bbox_a = a.bbox
    bbox_b = b.bbox
    return 0.5 * above(a, b) + 0.2 * get_proj_intersection(a, b) + 0.3 * near(a, b)


# Computes the "under" relation, which is taken to be symmetric to "over"
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def under(a, b):
    return on(b, a)


# Computes the "closer-than" relation
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def closer_than(a, b, pivot):
    return int(point_distance(a.centroid, pivot.centroid) < point_distance(b.centroid, pivot.centroid))


# Computes the deictic version of the "in-front-of" relation
# For two objects, one is in front of another iff it's closer and
# between the observer and that other object
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def in_front_of_deic(a, b):
    # def in_front_of_extr(a, b, observer):
    bbox_a = a.bbox
    max_dim_a = max(bbox_a[7][0] - bbox_a[0][0],
                    bbox_a[7][1] - bbox_a[0][1],
                    bbox_a[7][2] - bbox_a[0][2]) + 0.0001
    dist = get_distance_from_line(world.get_observer().centroid, b.centroid, a.centroid)
    # print ("{}, {}, CLOSER: {}, WC_DEIC: {}, WC_EXTR: {}, DIST: {}".format(a.name, b.name, closer_than(a, b, observer), within_cone(b.centroid - observer.centroid, a.centroid - observer.centroid, 0.95), within_cone(b.centroid - a.centroid, Vector((0, -1, 0)) - a.centroid, 0.8), e ** (- 0.1 * get_centroid_distance_scaled(a, b))))
    # print ("WITHIN CONE:")
    a_bbox = get_2d_bbox(vp_project(a, world.get_observer()))
    b_bbox = get_2d_bbox(vp_project(b, world.get_observer()))
    a_center = projection_bbox_center(a_bbox)
    b_center = projection_bbox_center(b_bbox)
    dist = np.linalg.norm(a_center - b_center)
    scaled_proj_dist = dist / (max(get_2d_size(a_bbox), get_2d_size(b_bbox)) + 0.001)
    # scaled_proj_dist = gaussian(scaled_proj_dist, 0, 1)

    # print ("BBOX :", a_bbox, b_bbox)
    # print ("PROJ DIST:" ,scaled_proj_dist)
    a_dist = np.linalg.norm(a.location - world.observer.location)
    b_dist = np.linalg.norm(b.location - world.observer.location)
    # print ("SIGM, OVERLAP :  ", sigmoid(b_dist - a_dist, 1, 0.5), math.e ** (-0.5 * scaled_proj_dist))
    return 0.5 * (sigmoid(b_dist - a_dist, 1, 0.5) + math.e ** (-0.5 * scaled_proj_dist))
    # return closer_than(a, b, world.observer) * math.e ** (-0.1 * scaled_dist)
    # return math.e ** (- 0.01 * get_centroid_distance_scaled(a, b)) * within_cone(b.centroid - a.centroid, world.front_axis, 0.7)
    # return math.e ** (- 0.01 * get_centroid_distance_scaled(a, b)) * within_cone(b.centroid - a.centroid, Vector((1, 0, 0)), 0.7)
    '''0.3 * closer_than(a, b, observer) + \
				  0.7 * (max(within_cone(b.centroid - observer.centroid, a.centroid - observer.centroid, 0.95),
				  within_cone(b.centroid - a.centroid, Vector((1, 0, 0)), 0.7)) * \
				  e ** (- 0.2 * get_centroid_distance_scaled(a, b)))#e ** (-dist / max_dim_a))'''


def in_front_of_extr(a, b):
    # proj_dist = math.fabs(world.front_axis.dot(a.location)) - math.fabs(world.front_axis.dot(b.location))
    # proj_dist_scaled = proj_dist / max(a.size, b.size)
    # print ("PROJ_DISTANCE", proj_dist_scaled)
    return math.e ** (- 0.01 * get_centroid_distance_scaled(a, b)) * within_cone(b.centroid - a.centroid,
                                                                                 -world.front_axis, 0.7)


# return sigmoid(proj_dist_scaled, 1, 1)

def in_front_of(a, b):
    if a == b:
        return 0
    front_deic = in_front_of_deic(a, b)
    front_extr = in_front_of_extr(a, b)
    # print ("IN_FRONT_OF: ", a, b, front_deic, front_extr)
    return max(front_deic, front_extr)


# Enable SVA
# Computes the deictic version of the "behind" relation
# which is taken to be symmetric to "in-front-of"
# Inputs: a, b - entities
# Return value: real number from [0, 1]
# def behind_deic(a, b):
#    return in_front_of_deic(b, a)

def behind(a, b):
    return in_front_of(b, a)


# Computes the "at" relation
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def at(a, b):
    if a == b:
        return 0
    return at_same_height(a, b) * touching(a, b) if touching(a, b) > 0.9 else at_same_height(a, b) * near(a, b)


def inside(a, b):
    a_bbox = a.bbox
    b_bbox = b.bbox
    shared_volume = get_bbox_intersection(a, b)
    proportion = shared_volume / b.volume
    return sigmoid(proportion, 1.0, 1.0)


# Computes the "touching" relation
# Two entities are touching each other if they
# are "very close"
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def touching(a, b):
    if a == b:
        return 0
    bbox_a = a.bbox
    bbox_b = b.bbox
    center_a = a.bbox_centroid
    center_b = b.bbox_centroid
    rad_a = max(bbox_a[7][0] - bbox_a[0][0], \
                bbox_a[7][1] - bbox_a[0][1], \
                bbox_a[7][2] - bbox_a[0][2]) / 2
    rad_b = max(bbox_b[7][0] - bbox_b[0][0], \
                bbox_b[7][1] - bbox_b[0][1], \
                bbox_b[7][2] - bbox_b[0][2]) / 2
    print(a, b)
    '''for point in bbox_a:
		if point_distance(point, center_b) < rad_b:
			return 1
	for point in bbox_b:
		if point_distance(point, center_a) < rad_a:
			return 1'''
    mesh_dist = 1e9
    planar_dist = 1e9
    shared_volume = shared_volume_scaled(a, b)
    # print ("SHARED VOLUME:", shared_volume)
    if b.get("planar") is not None:
        planar_dist = get_planar_distance_scaled(b, a)
    elif a.get("planar") is not None:
        planar_dist = get_planar_distance_scaled(a, b)
    # print ("PLANAR DIST: ", planar_dist)
    if get_centroid_distance_scaled(a, b) <= 1.5:
        # mesh_dist = closest_mesh_distance_scaled(a, b)
        mesh_dist = closest_mesh_distance(a, b) / (min(a.size, b.size) + 0.01)
    # print ("MESH DIST: ", mesh_dist)
    mesh_dist = min(mesh_dist, planar_dist)
    # print ("MESH DIST: ", mesh_dist)
    touch_face = 0
    for face in b.faces:
        for v in a.vertex_set:
            touch_face = max(is_in_face(v, face), touch_face)
    # print("MIN FACE DIST: ", min_face_dist)
    # print ("SHORTEST MESH DIST:" , mesh_dist)
    if shared_volume == 0:
        if touch_face > 0.95:
            ret_val = touch_face
        elif mesh_dist < 0.1:
            ret_val = math.exp(- mesh_dist)
        else:
            ret_val = math.exp(- 2 * mesh_dist)
    else:
        print(0.3 * math.exp(- 2 * mesh_dist) + 0.7 * (shared_volume > 0))
        ret_val = 0.3 * math.exp(- 2 * mesh_dist) + 0.7 * (shared_volume > 0)
    print("Touching " + a.name + ", " + b.name + ": " + str(ret_val))
    return ret_val


# Computes a special function that takes a maximum value at cutoff point
# and decreasing to zero with linear speed to the left, and with exponetial speed to the right
# Inputs: x - position; cutoff - maximum point; left, right - degradation coeeficients for left and
# right sides of the function
# Return value: real number from [0, 1]
def asym_inv_exp(x, cutoff, left, right):
    return math.exp(- right * math.fabs(x - cutoff)) if x >= cutoff else max(0, left * (x / cutoff) ** 3)


# Symmetric to the asym_inv_exp.
# Computes a special function that takes a maximum value at cutoff point
# and decreasing to zero with linear speed to the RIGHT, and with exponetial speed to the LEFT
# Inputs: x - position; cutoff - maximum point; left, right - degradation coeeficients for left and
# right sides of the function
# Return value: real number from [0, 1]
def asym_inv_exp_left(x, cutoff, left, right):
    return math.exp(- left * (x - cutoff) ** 2) if x < cutoff else max(0, right * (x / cutoff) ** 3)


def to_the_left_of(figure, ground=None):
    if ground is None:
        ground = world.entities

    return RightOf.compute(ground, figure)


# Computes the deictic version of to-the-left-of relation
# Inputs: a, b - entities
# Return value: real number from [0, 1]
def to_the_left_of_deic(a, b):
    return RightOf.compute(b, a)


def above(a, b):
    """Computes the 'a above b' relation, returns the certainty value.

	Parameters:
	a, b - objects of type Entity

	Return value:
	float value from [0, 1]
	"""
    # bbox_a = a.bbox
    # bbox_b = b.bbox
    # span_a = a.get_span()
    # span_b = b.get_span()
    # center_a = a.get_bbox_centroid()
    # center_b = b.get_bbox_centroid()
    # scaled_vertical_distance = (center_a[2] - center_b[2]) / ((span_a[5] - span_a[4]) + (span_b[5] - span_b[4]))
    # print ("ABOVE:", sigmoid(a.centroid[2] - b.centroid[2], 1, 0.1))
    # print ("CONE:",within_cone(a.centroid - b.centroid, Vector((0, 0, 1)), 0.05))
    vertical_dist_scaled = (a.centroid[2] - b.centroid[2]) / (max(a.dimensions[2], b.dimensions[2]) + 0.01)
    # print ("WITHIN CONE: ", a, within_cone(a.centroid - b.centroid, np.array([0, 0, 1.0]), 0.1), sigmoid(vertical_dist_scaled, 1, 3), vertical_dist_scaled)
    return within_cone(a.centroid - b.centroid, np.array([0, 0, 1.0]), 0.1) * sigmoid(vertical_dist_scaled, 1,
                                                                                      3)  # math.e ** (- 0.01 * get_centroid_distance_scaled(a, b))


def below(a, b):
    """Computes the 'a below b' relation, returns the certainty value.

	Parameters:
	a, b - objects of type Entity

	Return value:
	float value from [0, 1]
	"""
    return above(b, a)


# STUB
# def in_front_of_intr(a, b):
#    pass

# STUB
# def behind_intr(a, b):
#    in_front_of_intr(b, a)

def same_oriented(a, b):
    a_fr = a.front
    b_fr = b.front
    angle = math.fabs(np.dot(a_fr, b_fr))
    ret_val = math.e ** (- 1.5 * (angle * (1 - angle)))
    print("ORIENTATION: ", ret_val)
    return ret_val


def facing(a, b):
    a_fr = a.front
    b_fr = b.front
    centroid_disp = a.centroid - b.centroid
    centroid_disp /= np.linalg.norm(centroid_disp)
    a_angle = math.fabs(np.dot(a_fr, centroid_disp))
    b_angle = math.fabs(np.dot(b_fr, centroid_disp))
    a_facing = math.e ** (- 1.5 * (a_angle * (1 - a_angle)))
    b_facing = math.e ** (- 1.5 * (b_angle * (1 - b_angle)))
    ret_val = a_facing * b_facing
    # for bl in entities:
    #    if between(bl, a, b) > 0.8:

    print("FACING: ", a_facing, b_facing, ret_val)
    return ret_val


def clear(obj):
    """Return the degree to which the object obj is clear, i.e., has nothing on top."""
    ent_on = [on(entity, obj) for entity in entities if entity is not obj]
    return 1 - max(ent_on)


def higher_than_centroidwise(a, b):
    """Compute whether the centroid of a is higher than the centroid of b."""

    a0 = a.centroid
    b0 = b.centroid
    return sigmoid(a0[2] - b0[2], 1.0, 1.0)  # 1 / (1 + math.exp(-(a0[2] - b0[2]))


def higher_than(a, b):
    return higher_than_centroidwise(a, b)


def lower_than(a, b):
    return higher_than(b, a)


def taller_than(a, b):
    return a.dimensions[2] > b.dimensions[2]


def at_same_height(a, b):
    """
	Check if two entities are at the same height

	"""

    dist = np.linalg.norm(a.centroid[2] - b.centroid[2])
    scaled_dist = dist / (a.size + b.size + 0.01)
    return math.e ** (-scaled_dist)
    """a_higher_b = higher_than(a, b)
	b_higher_a = higher_than(b, a)
	if a_higher_b > 0.8 or b_higher_a > 0.8:
		return 0
	else:
		return 1 - a_higher_b * b_higher_a"""


def where(entity):
    entities = [ent for ent in world.active_context if ent.name != entity.name and ent.name != 'Table']
    entity_pairs = [(ent1, ent2) for (ent1, ent2) in list(itertools.combinations(world.active_context, r=2)) if
                    entity.name != ent1.name and entity.name != ent2.name and ent1.name != 'Table' and ent2.name != 'Table']

    def get_vals(pred_func):
        if pred_func != between:
            val = [((entity, ent), pred_func(entity, ent)) for ent in entities]
        else:
            val = [((entity, ent1, ent2), between(entity, ent1, ent2)) for (ent1, ent2) in entity_pairs]
        val.sort(key=lambda x: x[1], reverse=True)
        return val[0]

    # print ('WHERE PROC: ', entities, entity_pairs)
    max_val = 0
    ret_val = None

    val = get_vals(at)
    other_best = max([at(ent, val[0][1]) for ent in entities])
    # print ("NEXT TO: ", val, other_best, max_val)
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("next to", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(between)
    other_best = max([between(ent, val[0][1], val[0][2]) for ent in entities])
    # print ("BETWEEN: ", val, other_best, max_val)
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("between", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(on)
    other_best = max([on(ent, val[0][1]) for ent in entities])
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("on top of", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(under)
    other_best = max([under(ent, val[0][1]) for ent in entities])
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("under", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(to_the_left_of_deic)
    other_best = max([to_the_left_of_deic(ent, val[0][1]) for ent in entities])
    # print ("\nLEFT OF: ", val, other_best, max_val, [(ent, val[0][1], to_the_left_of_deic(ent, val[0][1])) for ent in entities], "\n")
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("to the left of", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(in_front_of_deic)
    other_best = max([in_front_of_deic(ent, val[0][1]) for ent in entities])
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("in front of", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(behind)
    other_best = max([behind(ent, val[0][1]) for ent in entities])
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("behind", val)

    if max_val > 0.9:
        return ret_val

    val = get_vals(to_the_right_of_deic)
    other_best = max([to_the_right_of_deic(ent, val[0][1]) for ent in entities])
    # print ("RIGHT OF: ", val, other_best, max_val)
    if val[1] > max_val and val[1] > other_best + 0.07:
        max_val = val[1]
        ret_val = ("to the right of", val)

    return ret_val


def superlative(predicate, entities, background):
    """Compute the "most" object from a given set of entities against a background."""

    # If the backgound is given, compare every entity against it and pick the max
    if background != None:
        result = max([(entity, predicate(entity, background)) for entity in entities if entity != background],
                     key=lambda x: x[1])[0]
    # If the is no background, e.g., for "topmost", just compare entities pairwise
    else:
        result = entities[0]
        if len(entities) > 1:
            for entity in entities[1:]:
                if predicate(entity, result) > predicate(result, entity):
                    result = entity
    return result


def extract_contiguous(entities):
    """
	Extract all the contiguous subsets of entities from the given set.

	Returns:
	A list of lists, where each inner list represents a contiguous subset of entities.
	"""

    if entities == []:
        return []

    groups = []

    # A flag marking if the given index has been processed and assigned a group.
    processed = [0] * len(entities)

    q = Queue()

    for idx in range(len(entities)):

        """
		If the current entity has not been assigned to a group yet,
		add it to the BFS queue and create a new group for it.
		"""
        if processed[idx] == 0:
            q.put(idx)
            processed[idx] = 1
            current_group = [entities[idx]]

            """
			Perform a BFS to find all the entities reachable from the one
			that originated the current group.
			"""
            while not q.empty():
                curr_idx = q.get()
                for idx1 in range(len(entities)):
                    # print (processed[idx1], entities[curr_idx], entities[idx1], touching(entities[curr_idx], entities[idx1]))
                    if processed[idx1] == 0 and touching(entities[curr_idx], entities[idx1]) > 0.85:
                        q.put(idx1)
                        processed[idx1] = 1
                        current_group.append(entities[idx1])

            groups.append(current_group)

    return groups


def get_region(region_type, region_mod, entity):
    x_max = entity.x_max
    x_min = entity.x_min
    y_max = entity.y_max
    y_min = entity.y_min
    z_max = entity.z_max
    z_min = entity.z_min
    dims = entity.dimensions

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    corners = np.array([[(x_min, y_min, 0)], [(x_min, y_max, 0)], [(x_max, y_max, 0)], [(x_max, y_min, 0)]])
    edges = np.array([
        # Front
        [corners[0], corners[3]],
        # Left
        [corners[0], corners[1]],
        # Back
        [corners[1], corners[2]],
        # Right
        [corners[2], corners[3]]])
    sides = np.array([
        # Left
        [(x_min, y_min, z_min), (x_min, y_max, z_min), (x_center, y_max, z_min), (x_center, y_min, z_min),
         (x_min, y_min, z_max), (x_min, y_max, z_max), (x_center, y_max, z_max), (x_center, y_min, z_max)],

        # Right
        [(x_center, y_min, z_min), (x_center, y_max, z_min), (x_max, y_max, z_min), (x_max, y_min, z_min),
         (x_center, y_min, z_max), (x_center, y_max, z_max), (x_max, y_max, z_max), (x_max, y_min, z_max)],

        # Front
        [(x_min, y_min, z_min), (x_min, y_center, z_min), (x_max, y_center, z_min), (x_max, y_min, z_min),
         (x_min, y_min, z_max), (x_min, y_center, z_max), (x_max, y_center, z_max), (x_max, y_min, z_max)],

        # Back
        [(x_min, y_center, z_min), (x_min, y_max, z_min), (x_max, y_max, z_min), (x_max, y_center, z_min),
         (x_min, y_center, z_max), (x_min, y_max, z_max), (x_max, y_max, z_max), (x_max, y_center, z_max)]])

    if region_type == "side":
        if region_mod == "left":
            return Entity(sides[0])
        elif region_mod == "right":
            return Entity(sides[1])
        elif region_mod == "front":
            return Entity(sides[2])
        elif region_mod == "back":
            return Entity(sides[3])
        else:
            return None

    if region_type == "edge":
        if region_mod == "front":
            return Entity(edges[0])
        elif region_mod == "left":
            return Entity(edges[1])
        elif region_mod == "back":
            return Entity(edges[2])
        elif region_mod == "right":
            return Entity(edges[3])
        else:
            return None
