import bpy
import bpy_types
import numpy as np
import math
from math import e, pi
import itertools
import os
import sys
import random
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from geometry_utils import *
import enum
import torch

class Entity(object):
	"""
	Comprises the implementation of the basic class representing relevant
	objects in the scene, such as primitives, composite structures, regions.
	"""

	scene = bpy.context.scene

	#Enumerates possible categories the entity object can belong to
	class Category(enum.Enum):
		PRIMITIVE = 0
		STRUCTURE = 1
		REGION = 2

	def __init__(self, components, name=None, explore_children=True):

		if type(components) == bpy_types.Object:
			components = [components]

		self.components = components
		self.full_mesh = []
		
		if name is not None:
			self.name = name
		else:
			self.name = self.components[0].name

		if explore_children == True:
			#Filling in the constituent objects starting with the parent
			queue = [item for item in self.components]
			while len(queue) != 0:
				parent = queue[0]
				if type(parent) != Entity:
					self.full_mesh.append(parent)
				queue.pop(0)
				for ob in Entity.scene.objects:                
					if ob.parent == parent and ob.type == "MESH":
						self.components.append(ob)												
						queue.append(ob)

		#print ("COMP: ", components, explore_children, type(self.components[0]))

		if len(self.components) == 1 and type(self.components[0]) == bpy_types.Object:
			self.category = self.Category.PRIMITIVE		

		elif len(self.components) > 1 and (type(self.components[0]) == Entity or type(self.components[0]) == bpy_types.Object): 			
			self.category = self.Category.STRUCTURE
		
			ent_components = []
			for comp in self.components:
				if len(comp.data.vertices) > 0:
					if type(comp) == Entity:
						ent_components.append(comp)
					else:
						ent_components.append(Entity(comp, explore_children=False))
			self.components = ent_components
			#self.components = [item for entity in components for item in entity.components]
			#self.components = [item for entity in components for item in entity.components]
			#self.name = 'struct=(' + '+'.join([item.name for item in self.components]) + ')'
			comp_names = [ent.name for ent in self.components if ent.components[0].get('main') is not None]
			if len(comp_names) == 1:
				self.name = comp_names[0]

		elif type(self.components[0]) == np.ndarray:
			self.category = self.Category.REGION

		#The type structure
		#Each object belong to hierarchy of types, e.g.,
		#Props -> Furniture -> Chair
		#This structure will be stored as a list
		#['Props', 'Furniture', 'Chair']
		self.type_structure = self.compute_type_structure()
		self.compute_geometry()
		
		#Color of the entity
		self.color_mod = self.get_color_mod()

	def compute_geometry(self):
		#Compute mesh-related data
		#self.vertex_set = self.compute_vertex_set()
		#self.faces = self.compute_faces()
		self.vertices, self.polygons = self.compute_mesh_geometry()
		# print (self.vertices[0], len(self.vertices))
		# print (self.vertex_set[0], len(self.vertex_set))
		#print (self.faces[0], len(self.faces))
		#print (self.polygons[0], len(self.polygons))		
		#print (self.name)
		#print (self.vertices)
		#print (self.polygons, "\n")

		'''
		print("\n" + self.name)
		for face in self.faces:
			if (len(face) == 4):
				v = get_distance_from_plane(face[3], face[0], face[1], face[2])
				if v>0.001:
					print(v)
		'''

		#Entity's mesh centroid
		self.centroid = self.compute_centroid()
		self.centroid_t = torch.tensor(self.centroid, dtype = torch.float32)
		self.location = self.centroid
		self.location_t = self.centroid_t  


		#self.bvh_trees = [BVHTree.FromObject(item, bpy.context.evaluated_depsgraph_get()) for item in self.full_mesh]
		#self.bvh_tree = BVHTree.FromPolygons(self.vertices, self.polygons, epsilon=0.0)
		# if self.category == self.Category.STRUCTURE:
		# 	print (self.name)
		# 	from voxeltree import Voxel
		# 	self.voxel_tree = Voxel(scope = {'vertices': self.vertices, 'edges':[], 'polygons': self.polygons}, depth = 5)
		self.compute_adjacent()

		#The coordiante span of the entity. In other words,
		#the minimum and maximum coordinates of entity's points
		self.span = self.compute_span()
		self.span_t = torch.tensor(self.span, dtype = torch.float32)

		#Separate values for the span of the entity, for easier access
		self.x_max = self.span[1]
		self.x_min = self.span[0]
		self.y_max = self.span[3]
		self.y_min = self.span[2]
		self.z_max = self.span[5]
		self.z_min = self.span[4]
		self.x_max_t = self.span_t[1]
		self.x_min_t = self.span_t[0]
		self.y_max_t = self.span_t[3]
		self.y_min_t = self.span_t[2]
		self.z_max_t = self.span_t[5]
		self.z_min_t = self.span_t[4]

		#The bounding box, stored as a list of triples of vertex coordinates
		self.bbox = self.compute_bbox()
		self.bbox_t = torch.tensor(self.bbox, dtype = torch.float32)

		#Bounding box's centroid
		self.bbox_centroid = self.compute_bbox_centroid()
		self.bbox_centroid_t = torch.tensor(self.bbox_centroid, dtype = torch.float32)


		#Dimensions of the entity in the format
		#[xmax - xmin, ymax - ymin, zmax - zmin]
		self.dimensions = self.compute_dimensions()
		self.dimensions_t = torch.tensor(self.dimensions, dtype = torch.float32)

		#The fundamental intrinsic vectors
		self.up = np.array([0, 0, 1])
		self.up_t = torch.tensor(self.up, dtype = torch.float32)

		self.front = np.array(self.components[0].get('frontal')) \
			if self.components[0].get('frontal') is not None else self.generate_frontal()
		#print ("FRONT: ", self.components[0].get('frontal'), self.generate_frontal(), self.front)
		if self.front is not None and len(self.front) == len(self.up):
			self.right = np.cross(self.front, self.up)
		else:
			self.right = None #np.array([0, -1, 0])

		# if self.category == self.Category.STRUCTURE:
		#  	print (self.name, self.type_structure, self.up, self.front, self.right)

		#self.canopy = self.get_canopy()

		self.radius = self.compute_radius()
		self.volume = self.compute_volume()
		self.size = self.compute_size()
		self.radius_t = torch.tensor(self.radius, dtype = torch.float32)
		self.volume_t = torch.tensor(self.volume, dtype = torch.float32)
		self.size_t = torch.tensor(self.size, dtype = torch.float32)
	   
		self.parent_offset = self.compute_parent_offset()
		self.ordering = self.induce_linear_order()

	def compute_adjacent(self):
		self.adj_data = {}
		for j in range(len(self.vertices)):
			neighbors = []
			for poly in self.polygons:
				for i in range(len(poly)):
					if j == poly[i]:
						if i > 0:
							neighbors.append(poly[i-1])
						else: 
							neighbors.append(poly[-1])
						if i < len(poly) - 1:
							neighbors.append(poly[i+1])
						else: 
							neighbors.append(poly[0])
						break
			self.adj_data[j] = neighbors

		#print (self.adj_data)

	   
	def set_type_structure(self, type_structure):
		self.type_structure = type_structure

	def compute_type_structure(self):
		"""Return the hierachy of types of the entity."""
		if self.category == self.Category.PRIMITIVE:
			if self.components[0].get('id') is not None:
				self.type_structure = self.components[0]['id'].split(".")
			else:
				self.type_structure = None
		elif self.category == self.Category.STRUCTURE:
			types = [comp.type_structure for comp in self.components if comp.type_structure is not None]
			if len(types) == 1:
				self.type_structure = types[0]
			else:
				self.type_structure = None
		return self.type_structure

	def compute_mesh_geometry(self):
		offset = 0
		vertices = []
		faces = []		
		for component in self.components:
			if type(component) == bpy_types.Object:
				world_matrix = self.components[0].matrix_world
				current_vert = []
				current_faces = []
				if hasattr(component.data, "vertices"):
					current_vert = [np.array(world_matrix @ v.co) for v in component.data.vertices]
				if hasattr(component.data, "polygons"):
					current_faces = [[idx + offset for idx in face.vertices] for face in component.data.polygons]
			else:
				current_vert = component.vertices
				current_faces = [[idx + offset for idx in poly] for poly in component.polygons]
			vertices += current_vert
			faces += current_faces
			offset += len(current_vert)
		return vertices, faces	

	def compute_span(self):
		"""Calculate the coordinate span of the entity."""
		if self.vertices != []:
			return [min([v[0] for v in self.vertices]),
					max([v[0] for v in self.vertices]),
					min([v[1] for v in self.vertices]),
					max([v[1] for v in self.vertices]),
					min([v[2] for v in self.vertices]),
					max([v[2] for v in self.vertices])]
		else:
			return [self.location, self.location, self.location, self.location, self.location, self.location]


	def compute_bbox(self):
		"""
		Calculate the bounding box of the entity
		and return it as an array of points.
		"""
		return [(self.span[0], self.span[2], self.span[4]),
			(self.span[0], self.span[2], self.span[5]),
			(self.span[0], self.span[3], self.span[4]),
			(self.span[0], self.span[3], self.span[5]),
			(self.span[1], self.span[2], self.span[4]),
			(self.span[1], self.span[2], self.span[5]),
			(self.span[1], self.span[3], self.span[4]),
			(self.span[1], self.span[3], self.span[5])]

	def compute_bbox_centroid(self):
		"""Compute and return the bounding box centroid."""
		return np.array([self.bbox[0][0] + (self.bbox[7][0] - self.bbox[0][0]) / 2,
						 self.bbox[0][1] + (self.bbox[7][1] - self.bbox[0][1]) / 2,
						 self.bbox[0][2] + (self.bbox[7][2] - self.bbox[0][2]) / 2])
   
	def compute_centroid(self):
		"""Compute and return the centroid the vertex set."""
		if self.vertices != []:
			return np.average(self.vertices, axis=0)
		else:
			#print ("LOCATION: ", self.components[0].name, self.components[0].location)
			return np.array(self.components[0].location)

	def compute_dimensions(self):
		"""Gets the dimensions of the entity as a list of number."""
		return [self.bbox[7][0] - self.bbox[0][0], self.bbox[7][1] - self.bbox[0][1], self.bbox[7][2] - self.bbox[0][2]]

	def compute_radius(self):
		"""Compute and return the radius of the circumscribed sphere of the entity."""
		if self.vertices != []:
			return max([np.linalg.norm(v - self.centroid) for v in self.vertices])
		else:
			return 0
		"""if not hasattr(self, 'radius'):
			total_mesh = self.get_total_mesh()
			centroid = self.get_centroid()
			self.radius = max([numpy.linalg.norm(v - centroid) for v in total_mesh])
		return self.radius"""
			
	def compute_volume(self):
		return (self.span[1] - self.span[0]) * (self.span[3] - self.span[2]) * (self.span[5] - self.span[4])

	def compute_parent_offset(self):
		"""Compute and return the offset of the entity relative to the location of its head object."""
		if self.category == self.Category.PRIMITIVE or self.category == self.Category.STRUCTURE:
			return self.components[0].location[0] - self.span[0], self.components[0].location[1] - self.span[2], self.components[0].location[2] - self.span[4]
		else:
			return None

	def get_color_mod(self):
		"""Returns the color of the entity."""
		if self.category == self.Category.PRIMITIVE and self.components[0].get('color_mod') is not None:
			return self.components[0]['color_mod'].lower()
		else:
			return None
		
	#Sets the direction of the longitudinal axis of the entity    
	def set_longitudinal(self, longitudinial):
		self.longitudinal = longitudinal

	#Sets the direction of the frontal axis of the entity
	def set_frontal(self, frontal):
		self.frontal = frontal

	def generate_frontal(self):	
		#print (self.name, self.type_structure)	
		types_fr = ['sofa', 'bookshelf', 'desk', 'tv', 'poster', 'picture',	'fridge', 'wall', 'book']
		types_nf = ['chair','table', 'bed', 'book', 'laptop', 'pencil', 'pencil holder', 'note', 'rose', 'vase', 'cardbox', 'box', 'ceiling light', \
			'lamp',	'apple','banana', 'plate', 'bowl', 'trash bin', 'trash can', 'ceiling fan', 'block', 'floor', 'ceiling']
		#print (self.type_structure)
		if self.name == 'Observer':
			return np.array([1, 0, 0])
		elif self.type_structure is None or len(self.type_structure) == 0 or self.type_structure[-1] in types_nf or self.type_structure[-2] in types_nf:
			return None
		elif self.type_structure[-1] in types_fr or self.type_structure[-2] in types_fr:
			if self.span[1] - self.span[0] >= self.span[3] - self.span[2]:
				extend = np.array([1, 0, 0])
				frontal = np.array([0, 1, 0])
			else:
				extend = np.array([0, 1, 0])
				frontal = np.array([0, 1, 0])
		else:
			normals = [get_normal(self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]) for face in self.polygons]
			normals = [item for item in normals if math.fabs(item[2]) < 0.5 * (math.fabs(item[0]) + math.fabs(item[1]))]
			if len(normals) > 0:
				frontal = np.average(np.array(normals), axis = 0)
			else:
				frontal = np.array([0, -1, 0])
		#print (self.name, self.category, frontal)
		return frontal

	#Checks if the entity has a given property
	def get(self, property):
		return self.components[0].get(property)

	#Coomputes the distance from a point to closest face of the entity
	def get_closest_face_distance(self, point):
		return min([get_distance_from_plane(point, self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]) for face in self.polygons])

	#STUB
	# def get_closest_distance(self, other_entity):
	# 	this_faces = self.get_faces()
	# 	other_faces = other_entity.get_faces()

	def print(self):
		print ("ENTITY: " + self.name)
		print ("\n".join([attr + ": " + self.__dict__[attr].__str__() for attr in self.__dict__.keys()]))

	def __str__(self):
		return "ENTITY: " + (self.name if self.name is not None else "NONE")

	def __repr__(self):
		return "ENT: " + (self.name if self.name is not None else "NONE")

	def induce_linear_order(self):        
		if self.category == self.Category.STRUCTURE:
			#print ("COMPUTING ORDER: ")
			centroid, direction, avg_dist, max_dist = fit_line([entity.centroid for entity in self.components])
			#print (centroid, direction, avg_dist, max_dist)
			if avg_dist < 0.7 and max_dist < 1:
				proj = [(entity, (entity.centroid - centroid).dot(direction)/(0.001 + np.linalg.norm(entity.centroid - centroid))) for entity in self.components]
				proj.sort(key = lambda x: x[1])                
				#print (proj)
				return [entity for (entity, val) in proj]
		return None

	def get_first(self):
		if self.ordering is not None and len(self.ordering) > 0:
			return self.ordering[0]
		else:
			return None

	def get_last(self):
		if self.ordering is not None and len(self.ordering) > 0:
			return self.ordering[-1]
		else:
			return None

	def compute_size(self):
		return self.radius

	def move_to(self, location=None, rotation=None):
		"""
		Move the entity into the location.

		"""

		displacement = location - self.location

		print ("COMP: ", self.components)
		for comp in self.components:
			#print ("LOC: ", comp.location, np.array(comp.location), location, displacement)
			#print ("VEC: ", Vector(np.array(comp.location) + displacement))
			comp.location = Vector(np.array(comp.location) + displacement)
		dg = bpy.context.evaluated_depsgraph_get().update()
		
		self.update()        

	def update(self):
		self.compute_geometry()

	def get_component_vectors(self):
		centroid = np.average([item.location for item in self.components])
		vectors = [item.location - centroid for item in self.components]
		return vectors

	def is_in_canopy(self, point):
		num_samples = 80
		total_hits = 0
		for i in range(num_samples):
			u = np.random.normal(0,1)
			v = np.random.normal(0,1)
			w = np.random.normal(0,1)
			vec = np.array([u, v, w])
			vec = vec / np.linalg.norm(vec)
			is_hit = False
			for comp in self.components:
				hit, loc, norm, face = comp.ray_cast(point, vec)
				is_hit = hit or is_hit
			if is_hit:
				total_hits += 1
		return float(total_hits) / num_samples

	def compute_canopy(self):
		x_min = 1e10
		x_max = -1e10
		y_min = 1e10
		y_max = -1e10
		z_min = 1e10
		z_max = -1e10
		for i in range(500):
			x = np.random.uniform(self.span[0], self.span[1])
			y = np.random.uniform(self.span[2], self.span[3])
			z = np.random.uniform(self.span[4], self.span[5])
			point = np.array([x, y, z])
			if self.is_in_canopy(point) > 0.8:
				print ("in canopy: ", point)
				x_min = min(x_min, point[0])
				y_min = min(y_min, point[1])
				z_min = min(z_min, point[2])
				x_max = max(x_max, point[0])
				y_max = max(y_max, point[1])
				z_max = max(z_max, point[2])
		print (x_min, x_max, y_min, y_max, z_min, z_max)
		if x_min < x_max and y_min < y_max and z_min < z_max:
			return np.array([x_min, x_max, y_min, y_max, z_min, z_max])
		else:
			return None

	def get_canopy(self):
		if self.components[0].get('canopy') is not None:
			return self.components[0]['canopy']
		else:
			canopy = self.compute_canopy()
			if canopy is not None:
				self.components[0]['canopy'] = canopy
				bpy.ops.wm.save_mainfile(filepath=bpy.data.filepath)
			return canopy

	def get_features(self):
		features = []
		for item in self.centroid:
			features.append(item)
		#features.append(self.centroid.tolist())
		for point in self.bbox:
			for coord in point:
				features.append(coord)
		#features.append(self.bbox)
		for item in self.span:
			features.append(item)
		#features.append(self.span)
		return features

	def build_octree(self):
		center = self.location


