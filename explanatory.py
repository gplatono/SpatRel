

class Explanatory:
	def __init__(self, spatial_source):
		self.spatial_source = spatial_source

	def explain(self, req):
		pass

	def explain_context(self, relation, args):
		tr = args[0]
		lms = args[1:]
		ranks = self.spatial_source.rank_tr(relation, lms)

		