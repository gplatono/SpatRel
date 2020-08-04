#import numpy as np
#import math
#import spatial
#from constraint_solver import *

from entity import Entity
from geometry_utils import *
from queue import Queue
#from mathutils import Vector
# import bpy_extras
#from functools import reduce
import itertools

world = None
observer = None

class Spatial:
	def __init__(self, world):
		self.world = world
		self.vis_proj = self.cache_2d_projections()
		self.axial_distances = self.pairwise_axial_distances()

		self.projection_intersection = ProjectionIntersection()
		self.within_cone_region = WithinConeRegion()
		self.frame_size = FrameSize()
		self.raw_distance = RawDistance()
		self.larger_than = LargerThan()
		self.closer_than = CloserThan()
		self.higher_than_centroidwise = HigherThan_Centroidwise()
		self.higher_than = HigherThan(connections = {'higher_than_centroidwise': self.higher_than_centroidwise})
		self.lower_than = LowerThan(connections = {'higher_than': self.higher_than})
		self.taller_than = TallerThan()
		self.at_same_height = AtSameHeight()
		self.supported = Supported()
		self.central = Central()
		self.horizontal_deictic_component = HorizontalDeicticComponent(network = self)
		self.vertical_deictic_component = VerticalDeicticComponent(network = self)
		self.touching = Touching()
		

		self.to_the_right_of_deictic = RightOf_Deictic(connections = {'horizontal_deictic_component': self.horizontal_deictic_component,
			'vertical_deictic_component': self.vertical_deictic_component})
		self.to_the_right_of_extrinsic = RightOf_Extrinsic()
		self.to_the_right_of_intrinsic = RightOf_Intrinsic()
		self.to_the_right_of = RightOf(connections = {'to_the_right_of_deictic': self.to_the_right_of_deictic,
			'to_the_right_of_intrinsic': self.to_the_right_of_intrinsic, 'to_the_right_of_extrinsic': self.to_the_right_of_extrinsic})
		self.to_the_left_of_deictic = LeftOf_Deictic(connections = {'to_the_right_of_deictic': self.to_the_right_of_deictic})
		self.to_the_left_of_extrinsic = LeftOf_Extrinsic(connections = {'to_the_right_of_extrinsic': self.to_the_right_of_extrinsic})
		self.to_the_left_of_intrinsic = LeftOf_Intrinsic(connections = {'to_the_right_of_intrinsic': self.to_the_right_of_intrinsic})
		self.to_the_left_of = LeftOf(connections = {'to_the_left_of_deictic': self.to_the_left_of_deictic,
			'to_the_left_of_intrinsic': self.to_the_left_of_intrinsic, 'to_the_left_of_extrinsic': self.to_the_left_of_extrinsic})
		self.in_front_of_deictic = InFrontOf_Deictic()
		self.in_front_of_extrinsic = InFrontOf_Extrinsic()
		self.in_front_of_intrinsic = InFrontOf_Intrinsic()
		self.in_front_of = InFrontOf(connections = {'in_front_of_deictic': self.in_front_of_deictic,
			'in_front_of_intrinsic': self.in_front_of_intrinsic, 'in_front_of_extrinsic': self.in_front_of_extrinsic})
		self.behind_deictic = Behind_Deictic(connections = {'in_front_of_deictic': self.in_front_of_deictic})        
		self.behind_extrinsic = Behind_Extrinsic(connections = {'in_front_of_extrinsic': self.in_front_of_extrinsic})
		self.behind_intrinsic = Behind_Intrinsic(connections = {'in_front_of_intrinsic': self.in_front_of_intrinsic})
		self.behind = Behind(connections = {'behind_deictic': self.behind_deictic,
			'behind_intrinsic': self.behind_intrinsic, 'behind_extrinsic': self.behind_extrinsic})
		self.above = Above(connections = {'within_cone_region': self.within_cone_region})
		self.below = Below(connections = {'above': self.above})
		self.near_raw = Near_Raw()
		self.near = Near(connections = {'near_raw': self.near_raw})
		self.over = Over(connections = {'above': self.above, 'projection_intersection': self.projection_intersection, 'near': self.near})
		self.on = On(connections = {'above': self.above, 'touching': self.touching, 'projection_intersection': self.projection_intersection, 
			'larger_than': self.larger_than})
		self.under = Under(connections = {'on': self.on})
		self.between = Between()
		self.inside = Inside()
		self.at = At(connections = {'at_same_height': self.at_same_height, 'touching': self.touching, 'near': self.near})

		self.str_to_pred = {
			'on.p': self.on,
			'on': self.on,

			'to_the_left_of.p': self.to_the_left_of,
			'to the left of': self.to_the_left_of,
			'left of': self.to_the_left_of,
			'left.a': self.to_the_left_of,
			'leftmost.a': self.to_the_left_of,
			'to_the_right_of.p': self.to_the_right_of,
			'to the right of': self.to_the_right_of,
			'right of': self.to_the_right_of,
			'right.a': self.to_the_right_of,
			'rightmost.a': self.to_the_right_of,
			'right.p': self.to_the_right_of,
			'left.p': self.to_the_left_of,

			'near.p': self.near,
			'near': self.near,
			'near_to.p': self.near,
			'close_to.p': self.near,
			'close.a': self.near,
			'next to': self.near,
			'beside': self.near,
			'besides': self.near,
			'on.p': self.on,
			'on_top_of.p': self.on,
			'on top of': self.on,
			'above.p': self.above,
			'above': self.above,
			'below.p': self.below,
			'below': self.below,

			'over.p': self.over,
			'over': self.over,
			'under.p': self.below,
			'under': self.below,
			'underneath.p': self.below,
			'supporting.p': self.under,

			'in.p': self.inside,			
			'inside.p': self.inside,
			'in': self.inside,
			'inside': self.inside,

			'touching.p': self.touching,
			'touching': self.touching,
			'touch.v': self.touching,
			'touch': self.touching,
			'adjacent_to.p': self.touching,

			'at.p': self.at,
			'next_to.p': self.at,
			'next to': self.at,

			'high.a': self.higher_than,
			'upper.a': self.higher_than,
			'highest.a': self.higher_than,
			'topmost.a': self.higher_than,
			'top.a': self.higher_than,
			'low.a': self.lower_than,
			'lowest.a': self.lower_than,

			'in_front_of.p': self.in_front_of,
			'in front of': self.in_front_of,
			'front.a': self.in_front_of,
			'frontmost.a': self.in_front_of,

			'behind.p': self.behind,
			'behind': self.behind,
			'backmost.a': self.behind,
			'back.a': self.behind,
			'farthest.a': self.behind,
			'far.a': self.behind,
			'between.p': self.between,
			'between': self.between,
			'in between': self.between,
			# 'clear.a': self.clear,
			# 'where.a': spatial.where,
			# 'exist.pred': exist,

			# 'face.v': spatial.facing,
			# 'facing.p': spatial.facing,

			# 'color.pred': color_pred,

			# 'blue.a': blue,
			# 'red.a': red,
			# 'green.a': green,

			# 'blue': blue,
			# 'red': red,
			# 'green': green,
			}

	def compute(self, relation, trs, lms):
		if relation not in self.str_to_pred:
			return -1        
		if len(lms) == 1:
			return self.str_to_pred[relation].compute(trs[0], lms[0])
		else:
			return self.str_to_pred[relation].compute(trs[0], lms[0], lms[1])

	def cache_2d_projections(self):
		proj = {}
		for ent in self.world.entities:
			proj[ent] = vp_project(ent, observer)
		return proj

	def pairwise_axial_distances(self):
		dist = {}
		for e1 in self.world.entities:
			for e2 in self.world.entities:
				if e1 != e2 and (e1, e2) not in dist:
					tr_bbox = get_2d_bbox(self.vis_proj[e1])
					lm_bbox = get_2d_bbox(self.vis_proj[e2])
					axial_dist = scaled_axial_distance(tr_bbox, lm_bbox)
					dist[(e1, e2)] = axial_dist
					dist[(e2, e1)] = axial_dist
		return dist

	def process_sample(self, annotation):
		relation = annotation[1]
		trs = [self.world.find_entity_by_name(annotation[0].strip())]
		lms = [self.world.find_entity_by_name(item.strip()) for item in annotation[2:]]
		sample = [trs[0]] + lms

		if 'not' in relation:
			label = 0
			relation = relation.replace('not ', '')
		else:
			label = 1

		relation = self.str_to_pred[relation].compute

		return sample, label, relation


class Node:
	def __init__(self, network=None, connections=None):
		self.arity = 2
		self.network = network
		self.set_connections(connections)        
		self.parameters = []

	def set_connections(self, connections):
		self.connections = connections

	def get_connections(self):
		return self.connections

	def get_parameters(self):
		return self.parameters

class ProjectionIntersection(Node):
	"""
	Computes the normalized area of the intersection of projection of two entities onto the XY-plane.
	Returns a real number from [0, 1].
	"""
	def compute(self, tr, lm):
		# bbox_tr = tr.bbox
		# bbox_lm = lm.bbox
		axmin = tr.span[0]
		axmax = tr.span[1]
		aymin = tr.span[2]
		aymax = tr.span[3]
		bxmin = lm.span[0]
		bxmax = lm.span[1]
		bymin = lm.span[2]
		bymax = lm.span[3]
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
	
	def str(self):
		return 'projection_intersection.n'

class WithinConeRegion(Node):
	"""
	Factor determining containment in an infinite cone-shaped beam
	with the given width, emanating from the origin object/point into a given
	direction.
	"""

	def __init__(self, ):        
		self.parameters = {'exponent_multiplier': 2}

	def compute(self, vect, direction, width):
		cos = direction.dot(vect) / (np.linalg.norm(direction) * np.linalg.norm(vect))
		angle = math.acos(cos)
		final_score = 1 / (1 + math.e ** (self.parameters['exponent_multiplier'] * (width - angle)))
		return final_score
   
	def str(self):
		return 'within_cone_region.p'

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
		for entity in world.entities:
			max_x = max(max_x, entity.span[1])
			min_x = min(min_x, entity.span[0])
			max_y = max(max_y, entity.span[3])
			min_y = min(min_y, entity.span[2])
			max_z = max(max_z, entity.span[5])
			min_z = min(min_z, entity.span[4])
		return max(max_x - min_x, max_y - min_y, max_z - min_z)
	
	def str(self):
		return 'frame_size.n'

class RawDistance(Node):

	def compute(self, tr, lm):
		dist = dist_obj(tr, lm)
		if tr.get('planar') is not None:
			dist = min(dist, get_planar_distance_scaled(tr, lm))
		elif lm.get('planar') is not None:
			dist = min(dist, get_planar_distance_scaled(lm, tr))
		elif tr.get('vertical_rod') is not None or tr.get('horizontal_rod') is not None or tr.get('rod') is not None:
			dist = min(dist, get_line_distance_scaled(tr, lm))
		elif lm.get('vertical_rod') is not None or lm.get('horizontal_rod') is not None or lm.get('rod') is not None:
			dist = min(dist, get_line_distance_scaled(lm, tr))
		elif tr.get('concave') is not None or lm.get('concave') is not None:
			dist = min(dist, closest_mesh_distance_scaled(tr, lm))        
		
		return dist
	
	def str(self):
		return 'raw_distance.n'

class LargerThan(Node):
	"""
	Computes whether the TR is larger than the LM.
	The result is a real number from [0, 1].
	"""

	def compute(self, tr, lm):       
		bbox_tr = tr.bbox
		bbox_lm = lm.bbox
		bbox_dim_diff = (bbox_lm[7][0] - bbox_lm[0][0] + bbox_lm[7][1] - bbox_lm[0][1] + bbox_lm[7][2] - bbox_lm[0][2]) \
			- (bbox_tr[7][0] - bbox_tr[0][0] + bbox_tr[7][1] - bbox_tr[0][1] + bbox_tr[7][2] - bbox_tr[0][2])
		return 1 / (1 + math.e ** bbox_dim_diff)
	def str(self):
		return 'larger_than.p'

class CloserThan(Node):
	"""
	Computes the "closer-than" relation. Returns a real number from [0, 1].
	"""

	def compute(self, tr, lm, pivot):
		return int(point_distance(tr.centroid, pivot.centroid) < point_distance(lm.centroid, pivot.centroid))

	def str(self):
		return 'closer_than.p'

class HigherThan_Centroidwise(Node):
	"""Compute whether the centroid of a is higher than the centroid of b."""
	def compute(self, tr, lm):
		return sigmoid(tr.centroid[2] - lm.centroid[2], 1.0, 1.0)

	def str(self):
		return 'higher_than_centroidwise.p'

class HigherThan(Node):
	def compute(self, tr, lm):
		return self.connections['higher_than_centroidwise'].compute(tr, lm)

	def str(self):
		return 'higher_than.p'

class LowerThan(Node):
	def compute(self, tr, lm):
		return self.connections['higher_than'].compute(lm, tr)

	def str(self):
		return 'lower_than.p'

class TallerThan(Node):
	def compute(self, tr, lm):
		return tr.dimensions[2] > lm.dimensions[2]

	def str(self):
		return 'taller_than.p'

class AtSameHeight(Node):
	def compute(self, tr, lm):
		"""
		Check if two entities are at the same height
		"""
		dist = np.linalg.norm(tr.centroid[2] - lm.centroid[2])
		scaled_dist = dist / (tr.size + lm.size + 0.01)
		return math.e ** (-scaled_dist)

	def str(self):
		return 'at_same_height.p'
	
class Supported(Node):
	"""
	Computes whether the TR is physically supported by the LM.
	The result is a real number from [0, 1].
	"""
	def __init__(self):
		self.parameters = {'rel_dist': 0.8}

	def compute(self, tr, lm):           
		direct_support = self.connections['touching'](tr, lm) * self.connections['above'](tr, lm)
		indirect_support = 0
		tr_h = tr.centroid[2]
		lm_h = lm.centroid[2]
		for entity in world.entities:
			e_h = entity.centroid[2]
			if (e_h - lm_h) / lm.size >= self.parameters['rel_dist'] and (tr_h - e_h) / tr.size >= self.parameters['rel_dist']:
				indirect_support = max(indirect_support, min(self.compute(tr, entity), self.compute(entity, lm)))
		#print ("SUPPORT: ", direct_support, indirect_support)
		return max(direct_support, indirect_support)

	def str(self):
		return 'supported_by.p'

class HorizontalDeicticComponent(Node):
	def compute(self, tr, lm):
		axial_dist = self.network.axial_distances[(tr, lm)]
		return asym_inv_exp(axial_dist[0], 1, 1, 0.05)

class VerticalDeicticComponent(Node):
	def compute(self, tr, lm):
		axial_dist = self.network.axial_distances[(tr, lm)]		
		return math.exp(- math.fabs(axial_dist[1]))

class Touching(Node):
	"""
	Computes the "touching" relation, where two entities are touching if they are "very close".
	The result is a real number from [0, 1]
	"""
	def compute(self, tr, lm):
		if tr == lm:
			return 0        
		mesh_dist = 1e9
		planar_dist = 1e9
		shared_volume = shared_volume_scaled(tr, lm)        
		if lm.get("planar") is not None:
			planar_dist = get_planar_distance_scaled(lm, tr)
		elif tr.get("planar") is not None:
			planar_dist = get_planar_distance_scaled(tr, lm)        
		if get_centroid_distance_scaled(tr, lm) <= 1.5:            
			mesh_dist = closest_mesh_distance(tr, lm) / (min(tr.size, lm.size) + 0.01)
		mesh_dist = min(mesh_dist, planar_dist)
		touch_face = 0

		#print ('INIT...', len(lm.faces))
		# for face in lm.faces:
		# 	for v in tr.vertex_set:
		# 		touch_face = max(is_in_face(v, face), touch_face)
		#print ('COMPLETE...')
		if shared_volume == 0:
			if touch_face > 0.95:
				ret_val = touch_face
			elif mesh_dist < 0.1:
				ret_val = math.exp(- mesh_dist)
			else:
				ret_val = math.exp(- 2 * mesh_dist)
		else:
			ret_val = 0.3 * math.exp(- 2 * mesh_dist) + 0.7 * (shared_volume > 0)
		#print ("Touching " + a.name + ", " + b.name + ": " + str(ret_val))
		return ret_val

	def str(self):
		return 'touching.p'

class RightOf_Deictic(Node):
	"""Deictic sense of the to-the-right-of relation."""

	def compute(self, tr, lm):
		horizontal_component = self.connections['horizontal_deictic_component'].compute(tr, lm)
		#asym_inv_exp(axial_dist[0], 1, 1, 0.05)
		vertical_component = self.connections['vertical_deictic_component'].compute(tr, lm)
		#math.exp(- math.fabs(axial_dist[1]))

		final_score = 0.4 * horizontal_component + 0.6 * vertical_component
		return final_score

	def str(self):
		return 'to_the_right_of_deictic.p'

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

	def str(self):
		return 'to_the_right_of_extrinsic.p'

class RightOf_Intrinsic(Node):
	"""Intrinsic sense of the to-the-right-of relation."""

	def compute(self, tr, lm):
		disp_vec = np.array(tr.bbox_centroid - lm.bbox_centroid)
		dist = np.linalg.norm(disp_vec)
		disp_vec = disp_vec / dist

		intrinsic_right = np.array(lm.right)
		print ('INTRINSIC: ', intrinsic_right, lm.right, disp_vec)
		cos = intrinsic_right.dot(disp_vec)

		final_score = math.e ** (- 0.1 * (1 - cos)) * math.e ** (- 0.05 * dist / max(tr.size, lm.size))
		return final_score

	def str(self):
		return 'to_the_right_of_intrinsic.p'

class RightOf(Node):
	"""Implementation of the general to-the-right-of relation."""

	def __init__(self, connections):
		self.set_connections(connections)
	
	def compute(self, tr, lm=None):
		ret_val = 0
		if type(tr) == Entity and type(lm) == Entity:
			connections = self.get_connections()
			deictic = connections['to_the_right_of_deictic'].compute(tr, lm)
			extrinsic = connections['to_the_right_of_extrinsic'].compute(tr, lm)			
			if lm.right is not None:
				intrinsic = connections['to_the_right_of_intrinsic'].compute(tr, lm)
			else:
				intrinsic = 0
			return max(deictic, extrinsic, intrinsic)
		elif lm is None:
			ret_val = np.average([self.compute(tr, entity) for entity in world.active_context])
		return ret_val

	def str(self):
		return 'to_the_right_of.p'


class LeftOf_Deictic(Node):
	def compute(self, tr, lm):
		return self.get_connections()['to_the_right_of_deictic'].compute(tr=lm, lm=tr)

	def str(self):
		return 'to_the_left_of_deictic.p'


class LeftOf_Extrinsic(Node):
	def compute(self, tr, lm):
		return self.get_connections()['to_the_right_of_extrinsic'].compute(tr=lm, lm=tr)
	def str(self):
		return 'to_the_left_of_extrinsic.p'

class LeftOf_Intrinsic(Node):
	def compute(self, tr, lm):
		return self.get_connections()['to_the_right_of_intrinsic'].compute(tr=lm, lm=tr)

	def str(self):
		return 'to_the_left_of_intrinsic.p'

class LeftOf(Node):
	def compute(self, tr, lm=None):
		ret_val = 0
		if type(tr) == Entity and type(lm) == Entity:
			connections = self.get_connections()
			deictic = connections['to_the_left_of_deictic'].compute(tr, lm)
			extrinsic = connections['to_the_left_of_extrinsic'].compute(tr, lm)			
			if tr.right is not None:
				intrinsic = connections['to_the_left_of_intrinsic'].compute(tr, lm)
			else:
				intrinsic = 0
			return max(deictic, extrinsic, intrinsic)

		elif lm is None:
			ret_val = np.average([self.compute(tr, entity) for entity in world.active_context])
		return ret_val

	def str(self):
		return 'to_the_left_of.p'

class InFrontOf_Deictic(Node):
	def compute(self, tr, lm):
		# def in_front_of_extr(a, b, observer):
		bbox_tr = tr.bbox
		max_dim_a = max(bbox_tr[7][0] - bbox_tr[0][0],
						bbox_tr[7][1] - bbox_tr[0][1],
						bbox_tr[7][2] - bbox_tr[0][2]) + 0.0001
		dist = get_distance_from_line(world.get_observer().centroid, lm.centroid, tr.centroid)
		a_bbox = get_2d_bbox(vp_project(tr, world.get_observer()))
		b_bbox = get_2d_bbox(vp_project(lm, world.get_observer()))
		a_center = projection_bbox_center(a_bbox)
		b_center = projection_bbox_center(b_bbox)
		dist = np.linalg.norm(a_center - b_center)
		scaled_proj_dist = dist / (max(get_2d_size(a_bbox), get_2d_size(b_bbox)) + 0.001)
		a_dist = np.linalg.norm(tr.location - world.observer.location)
		b_dist = np.linalg.norm(lm.location - world.observer.location)
		
		return 0.5 * (sigmoid(b_dist - a_dist, 1, 0.5) + math.e ** (-0.5 * scaled_proj_dist))

	def str(self):
		return 'in_front_of_deictic.p'
		
class InFrontOf_Extrinsic(Node):
	def compute(self, tr, lm):
		# proj_dist = math.fabs(world.front_axis.dot(a.location)) - math.fabs(world.front_axis.dot(b.location))
		# proj_dist_scaled = proj_dist / max(a.size, b.size)
		# print ("PROJ_DISTANCE", proj_dist_scaled)
		return math.e ** (- 0.01 * get_centroid_distance_scaled(tr, lm)) * within_cone(lm.centroid - tr.centroid,
																					 -world.front_axis, 0.7)
	def str(self):
		return 'in_front_of_extrinsic.p'

class InFrontOf_Intrinsic(Node):
	def compute(self, tr, lm):
		return math.e ** (- 0.01 * get_centroid_distance_scaled(tr, lm)) * within_cone(lm.centroid - tr.centroid,
																					   -lm.front, 0.7)
	def str(self):
		return 'in_front_of_intrinsic.p'

class InFrontOf(Node):
	def compute(self, tr, lm=None):
		ret_val = 0
		if type(tr) == Entity and type(lm) == Entity:
			connections = self.get_connections()
			deictic = connections['in_front_of_deictic'].compute(tr, lm)
			extrinsic = connections['in_front_of_extrinsic'].compute(tr, lm)
			if lm.front is not None:
				intrinsic = connections['in_front_of_intrinsic'].compute(tr, lm)
			else:
				intrinsic = 0

			return max(deictic, extrinsic, intrinsic)
		elif lm is None:
			ret_val = np.average([self.compute(tr, entity) for entity in world.active_context])
		return ret_val

	def str(self):
		return 'in_front_of.p'

class Behind_Deictic(Node):
	def compute(self, tr, lm):
		return self.get_connections()['in_front_of_deictic'].compute(tr=lm, lm=tr)

	def str(self):
		return 'behind_deictic.p'

class Behind_Extrinsic(Node):
	def compute(self, tr, lm):
		return self.get_connections()['in_front_of_extrinsic'].compute(tr=lm, lm=tr)

	def str(self):
		return 'behind_extrinsic.p'

class Behind_Intrinsic(Node):
	def compute(self, tr, lm):
		return self.get_connections()['in_front_of_intrinsic'].compute(tr=lm, lm=tr)

	def str(self):
		return 'behind_intrinsic.p'

class Behind(Node):
	def compute(self, tr, lm=None):
		ret_val = 0
		if type(tr) == Entity and type(lm) == Entity:
			connections = self.get_connections()
			return max(connections['in_front_of_deictic'].compute(tr, lm), connections['in_front_of_intrinsic'].compute(tr, lm),
					   connections['in_front_of_extrinsic'].compute(tr, lm))
		elif lm is None:
			ret_val = np.average([self.compute(tr, entity) for entity in world.active_context])
		return ret_val

	def str(self):
		return 'behind.p'

class Above(Node):

	def compute(self, tr, lm):
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
		vertical_dist_scaled = (tr.centroid[2] - lm.centroid[2]) / (max(tr.dimensions[2], lm.dimensions[2]) + 0.01)
		# print ("WITHIN CONE: ", a, within_cone(a.centroid - b.centroid, np.array([0, 0, 1.0]), 0.1), sigmoid(vertical_dist_scaled, 1, 3), vertical_dist_scaled)
		return self.connections['within_cone_region'].compute(tr.centroid - lm.centroid, np.array([0, 0, 1.0]), 0.1) \
				* sigmoid(vertical_dist_scaled, 1, 3)  # math.e ** (- 0.01 * get_centroid_distance_scaled(a, b))

	def str(self):
		return 'above.p'

class Below(Node):

	def compute(self, tr, lm):
		"""Computes the 'a below b' relation, returns the certainty value.

		Parameters:
		a, b - objects of type Entity

		Return value:
		float value from [0, 1]
		"""
		return self.connections['above'].compute(lm, tr)

	def str(self):
		return 'below.p'

class Near_Raw(Node):
	def compute(self, tr, lm):        
		bbox_tr = tr.bbox
		bbox_lm = lm.bbox
		dist = dist_obj(tr, lm)
		max_dim_a = max(bbox_tr[7][0] - bbox_tr[0][0],
						bbox_tr[7][1] - bbox_tr[0][1],
						bbox_tr[7][2] - bbox_tr[0][2])
		max_dim_b = max(bbox_lm[7][0] - bbox_lm[0][0],
						bbox_lm[7][1] - bbox_lm[0][1],
						bbox_lm[7][2] - bbox_lm[0][2])
		if tr.get('planar') is not None:
			# print ("TEST", a.name, b.name)
			dist = min(dist, get_planar_distance_scaled(tr, lm))
		elif lm.get('planar') is not None:
			dist = min(dist, get_planar_distance_scaled(tr, lm))
		elif tr.get('vertical_rod') is not None or tr.get('horizontal_rod') is not None or tr.get('rod') is not None:
			dist = min(dist, get_line_distance_scaled(tr, lm))
		elif lm.get('vertical_rod') is not None or lm.get('horizontal_rod') is not None or lm.get('rod') is not None:
			dist = min(dist, get_line_distance_scaled(lm, tr))
		elif tr.get('concave') is not None or lm.get('concave') is not None:
			dist = min(dist, closest_mesh_distance_scaled(tr, lm))

		fr_size = FrameSize().compute()
		raw_metric = math.e ** (- 0.1 * dist)
		'''0.5 * (1 - min(1, dist / avg_dist + 0.01) +'''
		#print("RAW NEAR: ", tr, lm, raw_metric * (1 - raw_metric / fr_size))
		return raw_metric * (1 - raw_metric / fr_size)

	def str(self):
		return 'near_raw.p'

class Near(Node):
	def compute(self, tr, lm=None):
		
		if tr == lm:
			return 0
		connections = self.get_connections()
		raw_near_measure = connections['near_raw'].compute(tr, lm)
		raw_near_tr = [connections['near_raw'].compute(tr, entity) for entity in world.entities if entity != tr]
		raw_near_lm = [connections['near_raw'].compute(lm, entity) for entity in world.entities if entity != lm]                
		avg_near = 0.5 * (np.average(raw_near_tr) + np.average(raw_near_lm))
	#     max_near_tr = max(raw_near_tr)
	#     max_near_lm = max(raw_near_lm)
	#     max_near = max(raw_near_measure, max_near_tr, max_near_lm)
	# # for entity in entities:
		#     if entity != tr and entity != lm:
		#         raw_near_tr = [connections['near_raw'].compute(tr, entity) for entity in entities if entity != tr and entity != lm]
		#         raw_near_lm = [connections['near_raw'].compute(lm, entity) for entity in entities if entity != tr and entity != lm]
		#         near_lm_entity = connections['near_raw'].compute(lm, entity)
		#         # print (entity.name, near_tr_entity, near_lm_entity)
		#         # if dist_a_to_entity < raw_dist:
		#          += [near_tr_entity]
		#         # if dist_b_to_entity < raw_dist:
		#         raw_near_lm += [near_lm_entity]


		# ratio = raw_near_measure / max_near
		# if (raw_near_measure < avg_near):
		#     near_measure_final = 0.5 * raw_near_measure
		# else:
		#     near_measure_final = raw_near_measure * ratio
		near_measure = raw_near_measure + (raw_near_measure - avg_near) * min(raw_near_measure, 1 - raw_near_measure)
		if tr.compute_size() > lm.compute_size():
			near_measure -= 0.1  # TODO
		elif tr.compute_size() < lm.compute_size():
			near_measure += 0.1
		return near_measure

	def str(self):
		return 'near.p'

class Between(Node):

	def __init__(self, connections=None):
		self.set_connections(connections)
		self.parameters = {'distance_scaling': -0.05}

	def compute(self, tr, lm1, lm2):
		#print("ENTERING THE BETWEEN...", a, b, c)
		# center_a = a.bbox_centroid
		# center_b = b.bbox_centroid
		# center_c = c.bbox_centroid
		#print (tr.name, lm1.name, lm2.name)

		tr_to_lm1 = lm1.centroid - tr.centroid
		tr_to_lm2 = lm2.centroid - tr.centroid

		cos = np.dot(tr_to_lm1, tr_to_lm2) / (np.linalg.norm(tr_to_lm1) * np.linalg.norm(tr_to_lm2) + 0.001)

		scaled_dist = np.linalg.norm(lm1.centroid - lm2.centroid) / (2 * tr.size)

		dist_coeff = math.exp(self.parameters['distance_scaling'] * scaled_dist)

		#print("BETWEEN DIST FACT: ", dist_coeff)
		ret_val = math.exp(- math.fabs(-1 - cos)) * dist_coeff
		return ret_val
	
	def str(self):
		return 'between.p'


class MetonymicOn(Node):
	def compute(self, tr, lm):
		on_val = 0
		on_cand = None
		for ob in lm.components:
			val = self.connections['on']
			
			
			# if ob.get('working_surface') is not None or ob.get('planar') is not None:
			# 	ret_val = max(ret_val, 0.5 * (v_offset(tr, ob_ent) + get_proj_intersection(tr, ob_ent)))
			# 	ret_val = max(ret_val, 0.5 * (int(near(tr, ob_ent) > 0.99) + larger_than(ob_ent, tr)))
		return ret_val

	def str(self):
		return 'on.p'

#Computes the "on" relation
#Inputs: a, b - entities
#Return value: real number from [0, 1]
class On(Node):
	def compute(self, tr, lm):
		if tr == lm:
			return 0
		proj_dist = np.linalg.norm(np.array([tr.location[0] - lm.location[0], tr.location[1] - lm.location[1]]))
		proj_dist_scaled = proj_dist / (max(tr.size, lm.size) + 0.01)
		#print ("LOCA: ", proj_dist_scaled)
		hor_offset = math.e ** (-0.3 * proj_dist_scaled)
		#print ("PROJ DIST: ", a, b, hor_offset)
		#print ("ON METRICS: ", touching(a, b), above(a, b), hor_offset, touching(a, b) * above(a, b) * hor_offset)
		ret_val =  self.connections['touching'].compute(tr, lm) * self.connections['above'].compute(tr, lm) \
							if hor_offset < 0.8 \
									else self.connections['above'].compute(tr, lm) #* touching(a, b)
		#print ("ON METRICS: ", touching(a, b), above(a, b), hor_offset, touching(a, b) * above(a, b) * hor_offset)
		#ret_val = max(ret_val, supporting(b, a))

		
		#print ("CURRENT ON: ", a, b, ret_val, above(a, b), touching(a, b), hor_offset)
	#    ret_val =  touching(a, b) * hor_offset if above(a, b) < 0.88 else above(a, b) * touching(a, b)        
		#print ("CURRENT ON:", ret_val)
		if lm.get('planar') is not None and self.connections['larger_than'].compute(lm, tr) and tr.centroid[2] > 0.5 * tr.dimensions[2]:
			ret_val = max(ret_val, touching(tr, lm))    
		#ret_val = 0.5 * (v_offset(a, b) + get_proj_intersection(a, b))
		#print ("ON {}, {}, {}".format(ret_val, get_proj_intersection(a, b), v_offset(a, b)))
		#ret_val = max(ret_val, 0.5 * (above(a, b) + touching(a, b)))
		#print ("ON {}".format(ret_val))
		for ob in lm.components:
			ob_ent = Entity(ob)
			if ob.get('working_surface') is not None or ob.get('planar') is not None:
				ret_val = max(ret_val, 0.5 * (v_offset(tr, ob_ent) + self.connections['projection_intersection'].compute(tr, ob_ent)))
				ret_val = max(ret_val, 0.5 * (int(near(tr, ob_ent) > 0.99) + self.connections['larger_than'].compute(ob_ent, tr)))
		if lm.get('planar') is not None and isVertical(lm):
			ret_val = max(ret_val, math.exp(- 0.5 * get_planar_distance_scaled(tr, lm)))
		return ret_val

	def str(self):
		return 'on.p'

class Over(Node):

	def compute(self, tr, lm):
		bbox_a = a.bbox
		bbox_b = b.bbox
		return 0.5 * self.connections['above'].compute(tr, lm) \
			+ 0.2 * self.connections['projection_intersection'].compute(tr, lm) \
			+ 0.3 * self.connections['near'].compute(tr, lm)

	def str(self):
		return 'over.p'

class Under(Node):
	"""
	Computes the "under" relation, which is taken to be symmetric to "over". 
	Returns a real number from [0, 1].
	"""
	def compute(self, tr, lm):
		return self.connections['on'].compute(lm, tr)
	
	def str(self):
		return 'under.p'

class At(Node):
	"""
	Computes the "at" relation. Returns a real number from [0, 1].
	"""
	def compute(self, tr, lm):
		if tr == lm:
			return 0
		touching = self.connections['touching'].compute(tr, lm)
		at_same_height = self.connections['at_same_height'].compute(tr, lm)
		return at_same_height * touching if touching > 0.9 else at_same_height * self.connections['near'].compute(tr, lm)

	def str(self):
		return 'next_to.p'

class Inside(Node):
	def compute(self, tr, lm):
		# a_bbox = a.bbox
		# b_bbox = b.bbox
		shared_volume = get_bbox_intersection(tr, lm)
		proportion = shared_volume / lm.volume
		return sigmoid(proportion, 1.0, 1.0)

	def str(self):
		return 'inside.p'


class Central(Node):
	def compute(self, tr, context=None):
		if context is None:
			context = self.network.world.active_context
		center = np.average([ent.centroid for ent in context])

		return math.exp(- 0.1 * np.linalg.norm(tr.centroid - center))

# =======================================================================================================
# ====================================OLD CODE STARTS HERE===============================================
# =======================================================================================================


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
