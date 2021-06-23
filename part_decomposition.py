import numpy as np
import heapq
from geometry_utils import *

class Decomposer:
	def __init__(self, mesh):
		self.mesh = mesh
		self.curvatures = {}

		self.mesh_curvature()
		print (self.curvatures)

		small = heapq.nlargest(4, self.curvatures.items(), key = lambda x: x[1])

		small = [(mesh[i[0]], i[1]) for i in small]
		print ("\nSMALLEST: ", small)
		
		# vects = {}
		# for p1 in mesh:

		# 	for p2 in mesh:
		# 		if p1 != p2:

	def is_inside(self, vertex):
		pass


	def curvature(self, vertex, neighborhood):
		vects = [neighbor - vertex for neighbor in neighborhood]
		print ("VALS: ", vertex, neighborhood)
		print ("VECTS: ", vects)
		return np.linalg.norm(np.average(vects, axis=0))

	def get_neighbors(self, vertex):
		neighbors = [(np.linalg.norm(v - vertex), tuple(v)) for v in self.mesh]

		# for v in self.mesh:
		# 	#print (v, vertex)
		# 	#norm = np.linalg.norm(v - vertex)
		# 	#print (norm)
		# 	if norm > 0.001:
		# 		heapq.heappush(neighbors, (norm, v))
		heapq.heapify(neighbors)

		#print (neighbors)
		ret_val = [item[1] for item in heapq.nsmallest(5, neighbors, key = lambda x: x[0])]
		#print (ret_val)
		return ret_val

	def mesh_curvature(self):
		for idx in range(len(self.mesh)):
			self.curvatures[idx] = self.curvature(self.mesh[idx], self.get_neighbors(self.mesh[idx]))