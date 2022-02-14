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
		self.geometry_content = {'vertices': [], 'edges': [], 'polygons': []}
		if root is not None:
			self.root = root
		else:
			self.root = self		

		if location is None:
			if type(scope) == dict:
				self.geometry_content = scope					
			else:
				offset = 0
				for entity in scope:
					self.geometry_content['vertices'] += entity.vertices
					self.geometry_content['polygons'] += [[idx + offset for idx in poly] for poly in entity.polygons]
					offset += len(entity.vertices)
			#print (self.geometry_content, len(self.geometry_content['vertices']))		

			x_min = y_min = z_min = 1e9
			x_max = y_max = z_max = -1e9
			for v in self.geometry_content['vertices']:
				x_min = min(x_min, v[0])
				y_min = min(y_min, v[1])
				z_min = min(z_min, v[2])
				x_max = max(x_max, v[0])
				y_max = max(y_max, v[1])
				z_max = max(z_max, v[2])
			# for entity in scope:
			# 	x_min = min(x_min, entity.x_min)
			# 	y_min = min(y_min, entity.y_min)
			# 	z_min = min(z_min, entity.z_min)
			# 	x_max = max(x_max, entity.x_max)
			# 	y_max = max(y_max, entity.y_max)
			# 	z_max = max(z_max, entity.z_max)
			location = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)])
			size = max([x_max - x_min, y_max - y_min, z_max - z_min])

			self.materials = []
			cnum = 25
			for i in range(cnum):
				cname = "col" + str(i)
				bpy.data.materials.new(name=cname)
				bpy.data.materials[cname].diffuse_color = (i/cnum, 0, 1-i/cnum, 0)				
				self.materials.append(bpy.data.materials[cname])
			
		else:
			self.geometry_content = scope


		if self == self.root:
			self.poly_points = [[self.geometry_content['vertices'][idx] for idx in poly] for poly in self.geometry_content['polygons']]
			self.poly_idx = list(range(len(self.poly_points)))
		else:
			self.poly_idx = self.geometry_content['polygons']
			
		self.location = location
		self.size = size
		self.depth = depth
		self.child_idx = child_idx
		self.material = None
		self.leaf_visual = None
		
		self.parent = parent
		self.children = [[[None, None], [None, None]], [[None, None], [None, None]]]
		#Order: X-up, X-down, Y-up, Y-down, Z-up, Z-down
		self.neighbors = [[None, None], [None, None], [None, None]]
		self.neighbors_linear = [None, None, None, None, None, None]
		self.is_leaf = True

		self.subdivide()
		self.node_count = self.get_node_count()
		# if self.is_leaf:
		# 	self.highlight()
		if depth == self.root.depth - 1:
			print (depth, len(self.poly_idx), self.node_count)
		
		self.counter = 0
		if self.root == self:
			self.postprocess()	
			print ("COUNTER", self.counter)

	def highlight_all(self):
		if self.is_leaf:
			self.highlight()
		else:
			self.run_on_children("highlight_all")


	#def is_concave(self):
	def print_all_neighbors(self):
		print("NODE: ", self.location, self.size, self.depth, self.neighbors_linear)
		self.run_on_children("print_all_neighbors")

	def postprocess(self):
		#self.fillNeighbors()
		#self.compute_all_NN()
		#self.highlight_all()
		bpy.context.evaluated_depsgraph_get().update()

	def is_peripheral(self):
		for neighbor in self.neighbors:
			if neighbor is None:
				return True
		return False

	def compute_NN(self, distance):
		# count = 0
		# count1 = 0
		# for neighbor1 in self.neighbors_linear:
		# 	if neighbor1 is not None:
		# 		count1 += 1
		# 		for neighbor2 in neighbor1.neighbors_linear:
		# 			if neighbor2 is not None 
		queue = [(self, distance)]
		visited = [self]
		count = 0

		while len(queue) > 0:			
			vox, dist = queue.pop(0)			
			count += 1
			if dist > 0:
				for neighbor in vox.neighbors_linear:
					if neighbor is not None and neighbor not in visited:
						queue.append((neighbor, dist-1))
						visited.append(neighbor)
		
		self.NN_count = count
		# if(self.NN_count > 4):
		# 	print (self.NN_count)

	def compute_all_NN(self, distance=2):
		self.compute_NN(distance)
		self.root.counter += 1
		#print (self.NN_count)
		self.run_on_children("compute_all_NN", distance)

	def run_on_children(self, method_name, *params):
		ret_val = []
		for i in range(2):
			for j in range(2):
				for k in range(2):
					if self.children[i][j][k] is not None:
						if len(params) == 0:
							ret_val += [getattr(self.children[i][j][k], method_name)()]
						else:
							ret_val += [getattr(self.children[i][j][k], method_name)(*params)]
					else:
						ret_val += [None]

		return ret_val

	def get_node_count(self):
		return 1 + sum([i for i in self.run_on_children("get_node_count") if i is not None])		

	def print_self(self):
		print ("\n", self, self.child_idx, self.location, self.size, self.parent, self.node_count)
		print (self.children)
		print (self.neighbors)

	def printStructure(self):
		self.print_self()
		self.run_on_children("printStructure")

	# def find_child_intersection(self, child_location):
	# 	intersect_list = []
	# 	for idx in self.poly_idx:			
	# 		if check_box_poly_intersection(child_location, self.size / 2, self.root.poly_points[idx]):
	# 			intersect_list.append(idx)
	# 	return intersect_list

	def subdivide(self):
		quart = self.size / 4
		if self.depth > 0:
			child_locations = np.array([(self.location[0] - quart, self.location[1] - quart, self.location[2] - quart), 
								(self.location[0] - quart, self.location[1] - quart, self.location[2] + quart),
								(self.location[0] - quart, self.location[1] + quart, self.location[2] - quart),
								(self.location[0] - quart, self.location[1] + quart, self.location[2] + quart),
								(self.location[0] + quart, self.location[1] - quart, self.location[2] - quart), 
								(self.location[0] + quart, self.location[1] - quart, self.location[2] + quart),
								(self.location[0] + quart, self.location[1] + quart, self.location[2] - quart),
								(self.location[0] + quart, self.location[1] + quart, self.location[2] + quart)])

			geom_data = [{}, {}, {}, {}, {}, {}, {}, {}]
			for i in range(len(geom_data)):
				geom_data[i] = {'vertices': [], 'edges': [], 'polygons': []}
			
			# for v in self.geometry_content['vertices']:
			# 	for i in range(8):
			# 		if check_box_point_containment(child_locations[i], self.size / 2, v):
			# 			geom_data[i]['vertices'].append(v)						
			# 			break
					
			for i in range(2):
				for j in range(2):
					for k in range(2):
						cidx = 4*i+2*j+k				
						intersect_list = [idx for idx in self.poly_idx if check_box_poly_intersection(child_locations[cidx], self.size / 2, self.root.poly_points[idx])]
						#self.find_child_intersection(child_locations[4*i+2*j+k])
						geom_data[cidx]['polygons'] = intersect_list
						if len(geom_data[cidx]['polygons']) > 0:							
							#geom_data = {'vertices': [], 'edges': [], 'polygons': intersect_list}							
							self.children[i][j][k] = Voxel(geom_data[cidx], child_locations[cidx], self.size / 2, self, self.root, self.depth-1, [i, j, k])
							self.is_leaf = False
			# self.children[0][0][0] = Voxel(self.intersect_list, child_locations[0], self.size / 2, self, self.root, self.depth-1, [0, 0, 0])
			# self.children[0][0][1] = Voxel(self.intersect_list, child_locations[1], self.size / 2, self, self.root, self.depth-1, [0, 0, 1])
			# self.children[0][1][0] = Voxel(self.intersect_list, child_locations[2], self.size / 2, self, self.root, self.depth-1, [0, 1, 0])
			# self.children[0][1][1] = Voxel(self.intersect_list, child_locations[3], self.size / 2, self, self.root, self.depth-1, [0, 1, 1])
			# self.children[1][0][0] = Voxel(self.intersect_list, child_locations[4], self.size / 2, self, self.root, self.depth-1, [1, 0, 0])
			# self.children[1][0][1] = Voxel(self.intersect_list, child_locations[5], self.size / 2, self, self.root, self.depth-1, [1, 0, 1])
			# self.children[1][1][0] = Voxel(self.intersect_list, child_locations[6], self.size / 2, self, self.root, self.depth-1, [1, 1, 0])
			# self.children[1][1][1] = Voxel(self.intersect_list, child_locations[7], self.size / 2, self, self.root, self.depth-1, [1, 1, 1])

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
			return self if check_box_point_containment(self.location, self.size, point) else None
	
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

		for i in range(2):
			for j in range(2):
				for k in range(2):
					if self.children[i][j][k] is not None:
						self.children[i][j][k].fillNeighbors()			

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
		#!!!
		return True
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
		block_mesh = bpy.data.meshes.new('Block_mesh')
		block = bpy.data.objects.new("voxx", block_mesh)
		bpy.context.collection.objects.link(block)

		#temp = self.NN_count / 25

		bm = bmesh.new()
		bmesh.ops.create_cube(bm, size=self.size)
		bm.to_mesh(block_mesh)
		bm.free()
		# if self.root.material is None:
		# 	bpy.data.materials.new(name="vox")
		# 	bpy.data.materials['vox'].diffuse_color = (temp, 0, 1-temp, 0)
		# 	self.root.material = bpy.data.materials['vox']
		# bpy.data.materials.new(name="vox")
		# bpy.data.materials['vox'].diffuse_color = (temp, 0, 1-temp, 0)

		block.data.materials.append(self.root.materials[self.NN_count])
		block.location = self.location				

if __name__ == "__main__":
	from world import World
	import time
	start_time = time.time()
	world = World(bpy.context.scene, simulation_mode=True)	
	vox = Voxel(scope = world.entities, depth=6)
	vox.print_all_neighbors()
	#vox.print_self()
	print (time.time() - start_time)

	# for idx1 in range(len(world.entities)):
	# 	for idx2 in range(idx1+1, len(world.entities)):
	# 		print (world.entities[idx1], world.entities[idx2], vox.contains([world.entities[idx1], world.entities[idx2]], depth=6))

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