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
from geometry_utils import *
import enum

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

	def __init__(self, components, name=None):

		#print (components, type(components))

		self.components = components
		self.name = name
		
		if type(components) == bpy_types.Object:
			self.category = self.Category.PRIMITIVE
			#Constituent objects
			#First object in the list is the parent or head object
			#Defining the entity
			main_object = components
			self.components = [main_object]            
			#self.constituents = [main_object]

			#Filling in the constituent objects starting with the parent
			queue = [main_object]
			while len(queue) != 0:
				parent = queue[0]
				queue.pop(0)
				for ob in Entity.scene.objects:                
					if ob.parent == parent and ob.type == "MESH":
						#self.constituents.append(ob)
						self.components.append(ob)
						queue.append(ob)
			#Name of the entity
			self.name = main_object.name

		elif type(components) == list and components != [] and type(components[0]) == Entity:
			self.category = self.Category.STRUCTURE
			self.components = [item for entity in components for item in entity.components]
			self.name = 'struct=(' + '+'.join([item.name for item in self.components]) + ')'
		elif type(components) == list or type(components) == np.ndarray:
			self.category = self.Category.REGION
			self.components = [components]

		#The type structure
		#Each object belong to hierarchy of types, e.g.,
		#Props -> Furniture -> Chair
		#This structure will be stored as a list
		#['Props', 'Furniture', 'Chair']
		self.type_structure = self.compute_type_structure()

		#Compute mesh-related data
		self.vertex_set = self.compute_vertex_set()
		self.faces = self.compute_faces()

		#The coordiante span of the entity. In other words,
		#the minimum and maximum coordinates of entity's points
		self.span = self.compute_span()

		#Separate values for the span of the entity, for easier access
		self.x_max = self.span[1]
		self.x_min = self.span[0]
		self.y_max = self.span[3]
		self.y_min = self.span[2]
		self.z_max = self.span[5]
		self.z_min = self.span[4]

		#The bounding box, stored as a list of triples of vertex coordinates
		self.bbox = self.compute_bbox()

		#Bounding box's centroid
		self.bbox_centroid = self.compute_bbox_centroid()

		#Dimensions of the entity in the format
		#[xmax - xmin, ymax - ymin, zmax - zmin]
		self.dimensions = self.compute_dimensions()

		#Entity's mesh centroid
		self.centroid = self.compute_centroid()

		self.location = self.centroid
	  
		#The fundamental intrinsic vectors
		self.up = np.array([0, 0, 1])
		self.front = np.array(self.components[0].get('frontal')) \
			if self.components[0].get('frontal') is not None else self.generate_frontal()
		#print ("FRONT: ", self.components[0].get('frontal'), self.generate_frontal(), self.front)
		if len(self.front) == len(self.up):
			self.right = np.cross(self.front, self.up)
		else:
			self.right = np.array([0, -1, 0])

		#print (self.name, self.front)

		self.radius = self.compute_radius()
		self.volume = self.compute_volume()
		self.size = self.compute_size()
	   
		#The parent offset
		self.parent_offset = self.compute_parent_offset()
		self.ordering = self.induce_linear_order()
		#Color of the entity
		self.color_mod = self.get_color_mod()

	def compute_geometry(self):
		#Compute mesh-related data
		self.vertex_set = self.compute_vertex_set()
		self.faces = self.compute_faces()

		#The coordiante span of the entity. In other words,
		#the minimum and maximum coordinates of entity's points
		self.span = self.compute_span()

		#Separate values for the span of the entity, for easier access
		self.x_max = self.span[1]
		self.x_min = self.span[0]
		self.y_max = self.span[3]
		self.y_min = self.span[2]
		self.z_max = self.span[5]
		self.z_min = self.span[4]

		#The bounding box, stored as a list of triples of vertex coordinates
		self.bbox = self.compute_bbox()

		#Bounding box's centroid
		self.bbox_centroid = self.compute_bbox_centroid()

		#Dimensions of the entity in the format
		#[xmax - xmin, ymax - ymin, zmax - zmin]
		self.dimensions = self.compute_dimensions()

		#Entity's mesh centroid
		self.centroid = self.compute_centroid()

		self.location = self.centroid
	  
		#The fundamental intrinsic vectors
		self.up = np.array([0, 0, 1])
		self.front = np.array(self.components[0].get('frontal')) \
			if self.components[0].get('frontal') is not None else self.generate_frontal()
		#print ("FRONT: ", self.components[0].get('frontal'), self.generate_frontal(), self.front)
		if len(self.front) == len(self.up):
			self.right = np.cross(self.front, self.up)
		else:
			self.right = np.array([0, -1, 0])

		#print (self.name, self.front)

		self.radius = self.compute_radius()
		self.volume = self.compute_volume()
		self.size = self.compute_size()
	   
		#The parent offset
		self.parent_offset = self.compute_parent_offset()
		self.ordering = self.induce_linear_order()	
	   
	def set_type_structure(self, type_structure):        
		self.type_structure = type_structure

	def compute_type_structure(self):
		"""Return the hierachy of types of the entity."""
		if self.category == self.Category.PRIMITIVE:
			if self.components[0].get('id') is not None:
				self.type_structure = self.components[0]['id'].split(".")
			else:
				self.type_structure = None
		else:
			self.type_structure = None
		return self.type_structure

	def compute_vertex_set(self):
		"""
		Compute and return the total vertex set of the entity.
		In case of a primitive or a structure it is the union 
		of the meshes of constituent objects.
		"""
		vertices = []
		if self.category == self.Category.PRIMITIVE:
			for obj in self.components:
				vertices += [obj.matrix_world @ v.co for v in obj.data.vertices]
			vertices = [np.array([v[0],v[1],v[2]]) for v in vertices]
		elif self.category == self.Category.STRUCTURE:
			vertices = [v for e in self.components for v in e.vertex_set]
		elif self.category == self.Category.REGION:
			vertices = self.components
		return vertices

	def compute_faces(self):
		"""Compute and return the list of faces of the entity."""
		faces = []
		#print ("CAT: ", self.category)
		if self.category == self.Category.PRIMITIVE:
			for ob in self.components:
				for face in ob.data.polygons:
					#print(ob.name, face)
					faces.append([ob.matrix_world @ ob.data.vertices[i].co for i in face.vertices])
		elif self.category == self.Category.STRUCTURE:
			faces = [f for entity in self.components for f in entity.faces]        
		return faces

	def compute_span(self):
		"""Calculate the coordinate span of the entity."""
		return [min([v[0] for v in self.vertex_set]),
				max([v[0] for v in self.vertex_set]),
				min([v[1] for v in self.vertex_set]),
				max([v[1] for v in self.vertex_set]),
				min([v[2] for v in self.vertex_set]),
				max([v[2] for v in self.vertex_set])]

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
		return sum(self.vertex_set) / len(self.vertex_set)
		"""centroid = Vector((0, 0, 0))
		vertex_count = 0
		
		for v in self.vertex_set():
			centroid += v
			vertex_count += 1
			centroid /= vertex_count
		self.centroid = centroid
		return self.centroid"""

	def compute_dimensions(self):
		"""Gets the dimensions of the entity as a list of number."""
		return [self.bbox[7][0] - self.bbox[0][0], self.bbox[7][1] - self.bbox[0][1], self.bbox[7][2] - self.bbox[0][2]]

	def compute_radius(self):
		"""Compute and return the radius of the circumscribed sphere of the entity."""
		return max([np.linalg.norm(v - self.centroid) for v in self.vertex_set])
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
			return self.components[0].location.x - self.span[0], self.components[0].location.y - self.span[2], self.components[0].location.z - self.span[4]
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
		#print ("FACES: ", self.faces)
		normals = [get_normal(face[0], face[1], face[2]) for face in self.faces]
		normals = [item for item in normals if math.fabs(item[2]) < 0.5 * (math.fabs(item[0]) + math.fabs(item[1]))]
		if len(normals) > 0:
			normals = np.average(np.array(normals), axis = 0)
		else:
			normals = np.array([0, -1, 0])
		#print ("NORMAL: ", self.name, normals)
		#for face in self.faces:
		#    normal = get_normal(face[0], face[1], face[2])
			#
		#    if math.fabs(normal[2]) < 0.3:
		#        return normal
		
		return normals

	#Checks if the entity has a given property
	def get(self, property):
		return self.components[0].get(property)

	#Coomputes the distance from a point to closest face of the entity
	def get_closest_face_distance(self, point):
		return min([get_distance_from_plane(point, face[0], face[1], face[2]) for face in self.faces])

	#STUB
	def get_closest_distance(self, other_entity):
		this_faces = self.get_faces()
		other_faces = other_entity.get_faces()

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

		
