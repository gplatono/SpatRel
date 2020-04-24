import bpy
import os
import sys
import numpy as np
import math
#from sklearn.metrics.pairwise import cosine_similarity

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

import geometry_utils
from entity import Entity

class Constructor(object):

	def __init__(self):
		pass

	def construct(self, args, rel_tuples):
		"""
		Combines the elements of args using a set of relations.

		Input: 
		args - list of entities, e.g., [Toyota, McDonalds, ...]
		rel_tuples - list of tuples of the form (arg0, relation, (arg1, {arg2})), where 
		arg0, arg, arg2 are elements of args and arg2 is optional, e.g., 
		[(Toyota, "on_top", (Texaco,)), (Texaco, "between", (Starbucks, McDonalds)), ...]

		"""

		pass

	def sample(self, relatum, relation, referent1=None, referent2=None):
		position = np.array([0, 0, 0])
		val = 0
		region_size = 5		
		iter = 0
		while val < 0.9:
			new_pos = np.random.multivariate_normal(mean = position, cov = (1 - val) * region_size * np.eye(3), size=1)[0]			
			relatum.move_to(new_pos)
			while (referent1 is not None and geometry_utils.intersection_entities(relatum, referent1)) or\
						(referent2 is not None and geometry_utils.intersection_entities(relatum, referent2)):
				new_pos = np.random.multivariate_normal(mean = position, cov = (1 - val) * region_size * np.eye(3), size=1)[0]			
				relatum.move_to(new_pos)
			curr_val = relation(relatum) if referent1 is None else relation(relatum, referent1) if referent2 is None else relation(relatum, referent1, referent2)
			if curr_val > val:
				val = curr_val
				position = new_pos
			iter += 1
			if iter == 500:
				return False
		print ("CONVERGED AFTER " + str(iter) + " ITERATIONS.")
		return True

	#returns magnitude of a 3d vector a.
	# def get_magnitude(self, a):
	# 	return math.sqrt((a[0] ** 2)+(a[1] ** 2)+(a[2] ** 2))

	#returns the dot products of two 3d vectors a and b.
	# def get_dot_product(self, a, b):
	# 	return ((a[0]*b[0])+(a[1]*b[1])+(a[2]*b[2]))

	#function that inputs two vectors a,b and returns "similarity" between the two.
  	#a1, a2, and c are scalers that have yet to be determined.
	def vectorSimilarity(self, a, b):
		w = 0.5
		c = 1
		mag_comp = w * math.e ** (-c * (math.fabs(np.linalg.norm(a) - np.linalg.norm(b))))
		dir_comp = (1 - w) * (geometry_utils.cosine_similarity(a,b) + 1) / 2
		print ("SIM: ", a, b, mag_comp, dir_comp)
		return mag_comp + dir_comp
		#w * math.e ** (-c * (math.fabs(np.linalg.norm(a) - np.linalg.norm(b)))) + (1 - w) * geometry_utils.cosine_similarity(a,b)

	def similarity(self, a, b):
		sim = self.structureSimilarity(a.get_component_vectors(), b.get_component_vectors())
		return sim

	# returns a value between 0 - 1 for the structure similarity 
	def structureSimilarity(self, a, b):
		max_len = max(len(a), len(b))		

		for i in range(max_len - len(b)):
			b.append((0,0,0))

		for i in range(max_len - len(a)):
			a.append((0,0,0))

		total_sim = 0
		for vectora in a:
			print (a, b)
			cand_sim = 0
			cand = 0
			for vectorb in b:
				curr_sim = self.vectorSimilarity(vectora, vectorb)
				#print ("SIM: ", vectora, vectorb, curr_sim)
				if(curr_sim > cand_sim):
					cand_sim = curr_sim
					cand = vectorb
			total_sim += cand_sim
			print (cand_sim, cand, vectora)
			b.remove(cand)

		return total_sim/max_len


constr = Constructor()
print(constr.structureSimilarity([(0, 0, 1), (0, 2, 0)], [(0, 0, 1), (0, 2, 0), (1, 0, 0)]))