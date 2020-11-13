import numpy as np
import bmesh
from geometry_utils import *
from mathutils.bvhtree import BVHTree

class Voxel:
	def __init__(self, scope, location=None, size=None, parent=None, depth=0, child_idx=[None, None, None]):
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

		self.location = location
		self.size = size
		print (self.location, self.size)
		self.depth = depth
		self.child_idx = child_idx
		self.compute_bbox()		
		self.intersect_list = []
		#temp_size = 0.01
		#increment = 
		#scope_copy = copy(scope)
		#while temp_size < self.size - 0.000001:				
		for entity in scope:
			if box_entity_vertex_containment(self.bbox_verts, entity):
				self.intersect_list.append(entity)
				continue
			for tree in entity.bvh_trees:
				print (entity, tree, tree.overlap(self.bvh_tree))
				if tree.overlap(self.bvh_tree) != []:
					self.intersect_list.append(entity)
					break

		#print (scope)
		print (self.intersect_list)

		self.parent = parent
		self.children = [[[None, None], [None, None]], [[None, None], [None, None]]]
		self.neighbors = [[None, None], [None, None], [None, None]]

		self.subdivide()
		self.fillNeighbors()


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
			self.children[0][0][0] = Voxel(self.intersect_list, loc[0], self.size / 2, self, self.depth-1, [0, 0, 0])
			self.children[0][0][1] = Voxel(self.intersect_list, loc[1], self.size / 2, self, self.depth-1, [0, 0, 1])
			self.children[0][1][0] = Voxel(self.intersect_list, loc[2], self.size / 2, self, self.depth-1, [0, 1, 0])
			self.children[0][1][1] = Voxel(self.intersect_list, loc[3], self.size / 2, self, self.depth-1, [0, 1, 1])
			self.children[1][0][0] = Voxel(self.intersect_list, loc[4], self.size / 2, self, self.depth-1, [1, 0, 0])
			self.children[1][0][1] = Voxel(self.intersect_list, loc[5], self.size / 2, self, self.depth-1, [1, 0, 1])
			self.children[1][1][0] = Voxel(self.intersect_list, loc[6], self.size / 2, self, self.depth-1, [1, 1, 0])
			self.children[1][1][1] = Voxel(self.intersect_list, loc[7], self.size / 2, self, self.depth-1, [1, 1, 1])

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
		self.bvh_tree = BVHTree.FromPolygons(self.bbox_verts, self.bbox_faces, epsilon=1.0)

	def findContainingVoxel(self, point, depth=-1):
		child_x = 0 if point[0] <= self.location[0] else 1
		child_y = 0 if point[1] <= self.location[1] else 1
		child_z = 0 if point[2] <= self.location[2] else 1
		
		if depth == 0 or self.children[child_x][child_y][child_z] is None:			
			return self if box_point_containment(self.bbox_verts, point) else None
	
		return self.children[child_x][child_y][child_z].findContainingVoxel(point, depth-1)

	def fillNeighbors(self):
		self.XDownAdjacent()		
		self.XUpAdjacent()		
		self.YDownAdjacent()		
		self.YUpAdjacent()		
		self.ZDownAdjacent()		
		self.ZUpAdjacent()		

	def XDownAdjacent(self):
		LCA = self
		while LCA.parent != None and self.child_idx[0] == 0:
			LCA = LCA.parent
		adjacent = LCA.findContainingVoxel(self.location - np.array([self.size, 0, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[0][0] = adjacent
			adjacent.neighbors[0][1] = self		

	def XUpAdjacent(self):
		LCA = self
		while LCA.parent != None and self.child_idx[0] == 1:
			LCA = LCA.parent
		adjacent = LCA.findContainingVoxel(self.location + np.array([self.size, 0, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[0][1] = adjacent
			adjacent.neighbors[0][0] = self		

	def YDownAdjacent(self):
		LCA = self
		while LCA.parent != None and self.child_idx[1] == 0:
			LCA = LCA.parent
		adjacent = LCA.findContainingVoxel(self.location - np.array([0, self.size, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[1][0] = adjacent
			adjacent.neighbors[1][1] = self		

	def YUpAdjacent(self):
		LCA = self
		while LCA.parent != None and self.child_idx[1] == 1:
			LCA = LCA.parent
		adjacent = LCA.findContainingVoxel(self.location + np.array([0, self.size, 0]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[1][1] = adjacent
			adjacent.neighbors[1][0] = self		

	def ZDownAdjacent(self):
		LCA = self
		while LCA.parent != None and self.child_idx[2] == 0:
			LCA = LCA.parent
		adjacent = LCA.findContainingVoxel(self.location - np.array([0, 0, self.size]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[2][0] = adjacent
			adjacent.neighbors[2][1] = self

	def ZUpAdjacent(self):
		LCA = self
		while LCA.parent != None and self.child_idx[2] == 1:
			LCA = LCA.parent
		adjacent = LCA.findContainingVoxel(self.location + np.array([0, 0, self.size]))
		if adjacent is not None and adjacent.size == self.size:
			self.neighbors[2][1] = adjacent
			adjacent.neighbors[2][0] = self		

class VoxelTree:
	def __init__(self, scene):
		self.scene = scene

	def build(self, levels):
		pass


filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)
from world import World
world = World(bpy.context.scene, simulation_mode=True)

Voxel(scope = world.entities)