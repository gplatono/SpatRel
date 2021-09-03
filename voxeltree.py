import numpy as np
import bmesh
from mathutils.bvhtree import BVHTree

import os
import sys
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from geometry_utils import *

class Voxel:
	def __init__(self, scope, location=None, size=None, parent=None, root=None, depth=0, child_idx=[None, None, None]):
		self.geometry_content = {'vertices': [], 'edges': [], 'polys': []}
		if root is not None:
			self.root = root
		else:
			self.root = self		

		if location is None:
			x_min = y_min = z_min = 1e9
			x_max = y_max = z_max = -1e9
			for entity in scope:
				x_min = min(x_min, entity.x_min)
				y_min = min(y_min, entity.y_min)
				z_min = min(z_min, entity.z_min)
				x_max = max(x_max, entity.x_max)
				y_max = max(y_max, entity.y_max)
				z_max = max(z_max, entity.z_max)
			location = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)])
			size = max([x_max - x_min, y_max - y_min, z_max - z_min])
			
			if type(scope) == dict:
				self.geometry_content = scope					
			else:
				offset = 0
				for entity in scope:
					self.geometry_content['vertices'] += entity.vertices
					self.geometry_content['polys'] += [[idx + offset for idx in poly] for poly in entity.polygons]
					offset += len(entity.vertices)
		else:
			self.geometry_content = scope
		self.location = location
		self.size = size
		self.depth = depth
		self.child_idx = child_idx
		self.compute_bbox()     
		self.intersect_list = []

		#if type(scope[0]) == Entity:
		for entity in scope:
			if box_entity_vertex_containment(self.bbox_verts, entity) or self.bvh_tree.overlap(entity.bvh_tree) != []:
				# if self.bvh_tree.overlap(entity.bvh_tree) != []:
				# 	print (self.location, self.size, entity, self.bvh_tree.overlap(entity.bvh_tree))
				self.intersect_list.append(entity)

		self.parent = parent
		self.children = [[[None, None], [None, None]], [[None, None], [None, None]]]
		self.neighbors = [[None, None], [None, None], [None, None]]
		self.neighbors_linear = [None, None, None, None, None, None]

		self.subdivide()
		self.node_count = self.get_node_count()
		if self.depth == 2:
			self.highlight()
		#self.fillNeighbors()

	def run_on_children(self, method_name):
		ret_val = []
		if self.children[0][0][0] is not None:
			ret_val += [getattr(self.children[0][0][0], method_name)(),
						getattr(self.children[0][0][1], method_name)(),
						getattr(self.children[0][1][0], method_name)(),
						getattr(self.children[0][1][1], method_name)(),
						getattr(self.children[1][0][0], method_name)(),
						getattr(self.children[1][0][1], method_name)(),
						getattr(self.children[1][1][0], method_name)(),
						getattr(self.children[1][1][1], method_name)()]

		return ret_val

	def get_node_count(self):
		return 1 + sum(self.run_on_children("get_node_count"))

		# if self.children[0][0][0] is not None:
		#   ret_val += self.children[0][0][0].get_size()
		#   ret_val += self.children[0][0][1].get_size()
		#   ret_val += self.children[0][1][0].get_size()
		#   ret_val += self.children[0][1][1].get_size()
		#   ret_val += self.children[1][0][0].get_size()
		#   ret_val += self.children[1][0][1].get_size()
		#   ret_val += self.children[1][1][0].get_size()
		#   ret_val += self.children[1][1][1].get_size()

		# return ret_val

	def print_self(self):
		print ("\n", self, self.child_idx, self.location, self.size, self.parent, self.node_count)
		print (self.children)
		print (self.neighbors)

	def printStructure(self):
		self.print_self()
		self.run_on_children("printStructure")

	def subdivide(self):
		quart = self.size / 4
		if self.depth > 0 and self.intersect_list != []:            
			child_locations = np.array([(self.location[0] - quart, self.location[1] - quart, self.location[2] - quart), 
								(self.location[0] - quart, self.location[1] - quart, self.location[2] + quart),
								(self.location[0] - quart, self.location[1] + quart, self.location[2] - quart),
								(self.location[0] - quart, self.location[1] + quart, self.location[2] + quart),
								(self.location[0] + quart, self.location[1] - quart, self.location[2] - quart), 
								(self.location[0] + quart, self.location[1] - quart, self.location[2] + quart),
								(self.location[0] + quart, self.location[1] + quart, self.location[2] - quart),
								(self.location[0] + quart, self.location[1] + quart, self.location[2] + quart)])            
			self.children[0][0][0] = Voxel(self.intersect_list, child_locations[0], self.size / 2, self, self.root, self.depth-1, [0, 0, 0])
			self.children[0][0][1] = Voxel(self.intersect_list, child_locations[1], self.size / 2, self, self.root, self.depth-1, [0, 0, 1])
			self.children[0][1][0] = Voxel(self.intersect_list, child_locations[2], self.size / 2, self, self.root, self.depth-1, [0, 1, 0])
			self.children[0][1][1] = Voxel(self.intersect_list, child_locations[3], self.size / 2, self, self.root, self.depth-1, [0, 1, 1])
			self.children[1][0][0] = Voxel(self.intersect_list, child_locations[4], self.size / 2, self, self.root, self.depth-1, [1, 0, 0])
			self.children[1][0][1] = Voxel(self.intersect_list, child_locations[5], self.size / 2, self, self.root, self.depth-1, [1, 0, 1])
			self.children[1][1][0] = Voxel(self.intersect_list, child_locations[6], self.size / 2, self, self.root, self.depth-1, [1, 1, 0])
			self.children[1][1][1] = Voxel(self.intersect_list, child_locations[7], self.size / 2, self, self.root, self.depth-1, [1, 1, 1])

	def compute_bbox(self):
		self.bbox_verts = np.array([(self.location[0] - self.size / 2, self.location[1] - self.size / 2, self.location[2] - self.size / 2), 
								(self.location[0] - self.size / 2, self.location[1] - self.size / 2, self.location[2] + self.size / 2),
								(self.location[0] - self.size / 2, self.location[1] + self.size / 2, self.location[2] - self.size / 2),
								(self.location[0] - self.size / 2, self.location[1] + self.size / 2, self.location[2] + self.size / 2),
								(self.location[0] + self.size / 2, self.location[1] - self.size / 2, self.location[2] - self.size / 2), 
								(self.location[0] + self.size / 2, self.location[1] - self.size / 2, self.location[2] + self.size / 2),
								(self.location[0] + self.size / 2, self.location[1] + self.size / 2, self.location[2] - self.size / 2),
								(self.location[0] + self.size / 2, self.location[1] + self.size / 2, self.location[2] + self.size / 2)])
		self.bbox_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [2, 3, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]]
		self.bvh_tree = BVHTree.FromPolygons(self.bbox_verts, self.bbox_faces, epsilon=0.0)

	def findContainingVoxel(self, point, depth=-1):
		child_x = 0 if point[0] <= self.location[0] else 1
		child_y = 0 if point[1] <= self.location[1] else 1
		child_z = 0 if point[2] <= self.location[2] else 1
		
		if depth == 0 or self.children[child_x][child_y][child_z] is None:          
			return self if box_point_containment(self.bbox_verts, point) else None
	
		return self.children[child_x][child_y][child_z].findContainingVoxel(point, depth-1)

	def fillNeighbors(self):
		if self.neighbors[0][0] is None:
			self.XDownAdjacent()
		if self.neighbors[0][1] is None:
			self.XUpAdjacent()  
		if self.neighbors[1][0] is None:
			self.YDownAdjacent()
		if self.neighbors[1][1] is None:
			self.YUpAdjacent()
		if self.neighbors[2][0] is None:
			self.ZDownAdjacent()
		if self.neighbors[2][1] is None:
			self.ZUpAdjacent()

		if self.children[0][0][0] is not None:
			self.children[0][0][0].fillNeighbors()
			self.children[0][0][1].fillNeighbors()
			self.children[0][1][0].fillNeighbors()
			self.children[0][1][1].fillNeighbors()
			self.children[1][0][0].fillNeighbors()
			self.children[1][0][1].fillNeighbors()
			self.children[1][1][0].fillNeighbors()
			self.children[1][1][1].fillNeighbors()

	def XDownAdjacent(self):
		if self.child_idx[0] == 1:
			adjacent = self.parent.children[0][self.child_idx[1]][self.child_idx[2]]
		else:
			LCA = self
			while LCA.child_idx[0] == 0:                
				LCA = LCA.parent            
			adjacent = LCA.findContainingVoxel(self.location - np.array([self.size, 0, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[0][0] = adjacent			
			adjacent.neighbors[0][1] = self
			self.neighbors_linear[0] = adjacent
			adjacent.neighbors_linear[1] = self

	def XUpAdjacent(self):
		if self.child_idx[0] == 0:
			adjacent = self.parent.children[1][self.child_idx[1]][self.child_idx[2]]
		else:
			LCA = self
			while LCA.child_idx[0] == 1:
				LCA = LCA.parent
			adjacent = LCA.findContainingVoxel(self.location + np.array([self.size, 0, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[0][1] = adjacent
			adjacent.neighbors[0][0] = self     
			self.neighbors_linear[1] = adjacent
			adjacent.neighbors_linear[0] = self

	def YDownAdjacent(self):
		if self.child_idx[1] == 1:
			adjacent = self.parent.children[self.child_idx[0]][0][self.child_idx[2]]
		else:
			LCA = self
			while LCA.child_idx[1] == 0:
				LCA = LCA.parent
			adjacent = LCA.findContainingVoxel(self.location - np.array([0, self.size, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[1][0] = adjacent			
			adjacent.neighbors[1][1] = self     
			self.neighbors_linear[2] = adjacent
			adjacent.neighbors_linear[3] = self

	def YUpAdjacent(self):
		if self.child_idx[1] == 0:
			adjacent = self.parent.children[self.child_idx[0]][1][self.child_idx[2]]
		else:
			LCA = self
			while LCA.child_idx[1] == 1:
				LCA = LCA.parent
			adjacent = LCA.findContainingVoxel(self.location + np.array([0, self.size, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[1][1] = adjacent
			adjacent.neighbors[1][0] = self     
			self.neighbors_linear[3] = adjacent
			adjacent.neighbors_linear[2] = self

	def ZDownAdjacent(self):
		if self.child_idx[2] == 1:
			adjacent = self.parent.children[self.child_idx[0]][self.child_idx[1]][0]
		else:
			LCA = self
			while LCA.child_idx[2] == 0:
				LCA = LCA.parent        
			adjacent = LCA.findContainingVoxel(self.location - np.array([0, 0, self.size]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[2][0] = adjacent
			adjacent.neighbors[2][1] = self
			self.neighbors_linear[4] = adjacent
			adjacent.neighbors_linear[5] = self

	def ZUpAdjacent(self):
		if self.child_idx[2] == 0:
			adjacent = self.parent.children[self.child_idx[0]][self.child_idx[1]][1]
		else:
			LCA = self
			while LCA.child_idx[2] == 1:
				LCA = LCA.parent
			adjacent = LCA.findContainingVoxel(self.location + np.array([0, 0, self.size]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[2][1] = adjacent
			adjacent.neighbors[2][0] = self
			self.neighbors_linear[5] = adjacent
			adjacent.neighbors_linear[4] = self

	def contains(self, entities, depth=-1):
		#print (self.location, self.size, self.intersect_list)
		for ent in entities:
			if ent not in self.intersect_list:
				return False

		if depth == 0 or self.children[0][0][0] is None:
			#print (self.location, self.size, self.bbox_verts)
			return True

		child_res = False
		child_res = child_res or self.children[0][0][0].contains(entities, depth-1)
		child_res = child_res or self.children[0][0][1].contains(entities, depth-1)
		child_res = child_res or self.children[0][1][0].contains(entities, depth-1)
		child_res = child_res or self.children[0][1][1].contains(entities, depth-1)
		child_res = child_res or self.children[1][0][0].contains(entities, depth-1)
		child_res = child_res or self.children[1][0][1].contains(entities, depth-1)
		child_res = child_res or self.children[1][1][0].contains(entities, depth-1)
		child_res = child_res or self.children[1][1][1].contains(entities, depth-1)

		return child_res

	def find_container(self, point, depth=-1):
		if depth == 0:
			if box_point_containment(self.bbox_verts, point):
				return self
			else:
				return None
		elif self.children[0][0][0] is None:
			if depth == -1:
				return self
			else:
				return None
		else:
			res = self.children[0][0][0].find_container(point, depth-1)
			if res is None:
				res = self.children[0][0][1].find_container(point, depth-1)
			if res is None:
				res = self.children[0][1][0].find_container(point, depth-1)
			if res is None:
				res = self.children[0][1][1].find_container(point, depth-1)
			if res is None:
				res = self.children[1][0][0].find_container(point, depth-1)
			if res is None:
				res = self.children[1][0][1].find_container(point, depth-1)
			if res is None:
				res = self.children[1][1][0].find_container(point, depth-1)
			if res is None:
				res = self.children[1][1][1].find_container(point, depth-1)
			return res

	def is_boundary(self):
		return self.neighbors[0][0] is None or \
				self.neighbors[0][1] is None or \
				self.neighbors[1][0] is None or \
				self.neighbors[1][1] is None or \
				self.neighbors[2][0] is None or \
				self.neighbors[2][1] is None

	def cut(self, voxel):	
		#dist = {}
		vox_dict = {}
		voxel.distance = 0

		#Put None to speed up checks in the BFS, i.e., avoid checking for non-None-ness of a neighbor
		vox_dict[None] = -1

		vox_dict[voxel] = 0		
		queue = [voxel]

		while len(queue) != 0:
			v = queue[0]
			queue.pop(0)			
			if v.neighbors[0][0] not in vox_dict:
				vox_dict[v.neighbors[0][0]] = vox_dict[v] + 1
				queue.append(v.neighbors[0][0])
			if v.neighbors[0][1] not in vox_dict:# and not hasattr(v.neighbors[0][0], 'distance'):
				vox_dict[v.neighbors[0][1]] = vox_dict[v] + 1
				queue.append(v.neighbors[0][1])				
			if v.neighbors[1][0] not in vox_dict:# and not hasattr(v.neighbors[0][0], 'distance'):
				vox_dict[v.neighbors[1][0]] = vox_dict[v] + 1
				queue.append(v.neighbors[1][0])
			if v.neighbors[1][1] not in vox_dict:
				vox_dict[v.neighbors[1][1]] = vox_dict[v] + 1
				queue.append(v.neighbors[1][1])
			if v.neighbors[2][0] not in vox_dict:
				vox_dict[v.neighbors[2][0]] = vox_dict[v] + 1
				queue.append(v.neighbors[2][0])
			if v.neighbors[2][1] not in vox_dict:
				vox_dict[v.neighbors[2][1]] = vox_dict[v] + 1
				queue.append(v.neighbors[2][1])

			if v.is_boundary():
				cut_flag = True
				for neighbor in v.neighbors_linear:
					if neighbor is not None and neighbor.is_boundary() and vox_dict[neighbor] <= vox_dict[v]:
						flag = False
						break
				if cut_flag:
					return v

	def find_cut(self, point):
		origin = self.find_container(point)
		target = self.cut(origin)
		return origin, target

	def highlight(self):
		# bpy.ops.mesh.primitive_cube_add()
		# block = bpy.context.selected_objects[0]
		# block.location = self.location
		# block.size = self.size
		# bpy.data.materials.new(name="vox")		
		# bpy.data.materials['vox'].diffuse_color = (1, 0, 0, 0)		
		# block.data.materials.append(bpy.data.materials['vox'])

		block_mesh = bpy.data.meshes.new('Block_mesh')
		block = bpy.data.objects.new("voxx", block_mesh)
		bpy.context.collection.objects.link(block)

		bm = bmesh.new()
		bmesh.ops.create_cube(bm, size=self.size)
		bm.to_mesh(block_mesh)
		bm.free()
		bpy.data.materials.new(name="vox")		
		bpy.data.materials['vox'].diffuse_color = (1, 0, 0, 0)		
		block.data.materials.append(bpy.data.materials['vox'])
		block.location = self.location		
		bpy.context.evaluated_depsgraph_get().update()		

if __name__ == "__main__":
	from world import World
	world = World(bpy.context.scene, simulation_mode=True)
	vox = Voxel(scope = world.entities, depth=7)
	vox.print_self()

	for idx1 in range(len(world.entities)):
		for idx2 in range(idx1+1, len(world.entities)):
			print (world.entities[idx1], world.entities[idx2], vox.contains([world.entities[idx1], world.entities[idx2]], depth=6))

	#vox.highlight()



# laptop = world.find_entity_by_name('laptop')
# table = world.find_entity_by_name('table')
# cardbox1 = world.find_entity_by_name('Cardbox 1')
# rbook1 = world.find_entity_by_name('Red Book 1')
# ybook1 = world.find_entity_by_name('Yellow Book 1')
# floor = world.find_entity_by_name('Floor')
# cardbox2 = world.find_entity_by_name('Cardbox 2')
#cube1 = world.find_entity_by_name('cube 1')
#cube2 = world.find_entity_by_name('cube 2')
#vox.fillNeighbors()
#print (cube1.bvh_tree.overlap(cube2.bvh_tree))
# print (vox.contains([laptop, table], depth=6))
# print (vox.contains([cardbox1, table], depth=6))
# print (vox.contains([cardbox2, table], depth=6))
# print (vox.contains([rbook1, ybook1], depth=6))
# print (vox.contains([cardbox2, floor], depth=6))