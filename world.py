import itertools
import time
import json
import requests
import numpy as np
import os
import datetime

import bpy
import bmesh
from mathutils import Vector
from mathutils import Quaternion
from entity import Entity
from geometry_utils import *

#import spatial

class World(object):
	"""
	Incapsulates the Blender scene and all the objects in it,
	i.e., a self-contained 'worldlet', as well as provides 
	the convenient API to access their properties, like the size
	of the world, object hierarchies, etc.
	"""
	def __init__(self, scene, simulation_mode=False):
		self.scene = scene
		self.entities = []
		self.active_context = []
		self.simulation_mode = simulation_mode

		#Set the fundamental extrinsic axes
		self.right_axis = np.array([1, 0, 0])
		self.front_axis = np.array([0, -1.0, 0])
		self.up_axis = np.array([0, 0, 1.0])
		#Sizes of BW objects in meters
		self.block_edge = 1.0
		#self.block_edge = 0.155
		self.table_edge = 1.53
		self.bw_multiplier = 1.0 / 0.155
		self.kinectLeft = (-0.75, 0.27, 0.6)
		self.kinectRight = (0.75, 0.27, 0.6)

		#List of  possible color modifiers
		self.color_mods = ['black', 'red', 'blue', 'brown', 'green', 'yellow']		

		#self.scene_setup()

		self.verbose = False
		self.verbose_rotation = False		
		
		bpy.utils.register_class(self.ProcessInputOp)
		bpy.utils.register_class(self.ModalTimerOp)
		self.ProcessInputOp.world = self
		self.ModalTimerOp.world = self

		bpy.ops.view3d.process_input()

		self.moved_blocks = []
		
		for obj in self.scene.objects:
			if obj.get('main') is not None and obj.get('enabled') is None:
				self.entities.append(Entity(obj))
				if self.entities[-1].name.lower() != "table":
					self.active_context.append(self.entities[-1])
		
		#Number of objects in the world
		self.N = len(self.entities)
		self.dimensions = self.get_dimensions()		
		self.observer = self.create_observer()

		#Create and save the initial state of the world
		self.history = []
		self.move_queue = []

		if self.simulation_mode == False:
			block_data = self.get_block_data()
			block_data.sort(key = lambda x : x[1][0])
			for idx in range(len(block_data)):
				id, location, rotation = block_data[idx]
				self.block_to_ids[self.blocks[idx]] = id
				self.block_by_ids[id] = self.blocks[idx]
				self.blocks[idx].location = location
				self.blocks[idx].rotation_euler = rotation

			bpy.ops.wm.modal_timer_operator()	
		else:
			# self.history.append(self.State(self.entities))
			self.record_history()
			if len(self.history) == 1:
				self.make_checkpoint()

		self.init_event_log()	
		self.start_time = time.time()	

	def get_block_data(self):
		url = "http://127.0.0.1:1236/world-api/block-state.json"
		response = requests.get(url, data="")
		json_data = json.loads(response.text)
		block_data = []        

		for segment in json_data['BlockStates']:
			position = self.bw_multiplier * np.array([round(float(x), 2) for x in segment['Position'].split(",")])
			rotation = Quaternion([round(float(x), 2) for x in segment['Rotation'].split(",")]).to_euler()
			block_data.append((segment['ID'], position, rotation))            

		return block_data
		
	def get_observer(self):
		if not hasattr(self, 'observer') or self.observer == None:
			self.observer = self.create_observer()
		return self.observer

	def create_observer(self):
		"""Create and configure the special "observer" object
		(which is just a camera). Needed for deictic relations as
		well as several other aspects requiring the POV concept,
		e.g., taking screenshots.
		"""
	
		lamp = bpy.data.lights.new(name="Lamp", type = 'POINT')
		lamp.energy = 30		

		if bpy.data.objects.get("Lamp") is not None:
			lamp_obj = bpy.data.objects["Lamp"]
		else:
			lamp_obj = bpy.data.objects.new("Lamp", lamp)			
			bpy.context.collection.objects.link(lamp_obj)
			
		cam = bpy.data.cameras.new("Camera")
		if bpy.data.objects.get("Camera") is not None:
			cam_ob = bpy.data.objects["Camera"]
		else:
			cam_ob = bpy.data.objects.new("Camera", cam)
			bpy.context.collection.objects.link(cam_ob)
			
		#lamp_obj.location = (-20, 0, 10)
		#cam_ob.location = (-15.5, 0, 7)
		lamp_obj.location = (0, -20, 10)
		cam_ob.location = (0, -9, 3)
		cam_ob.rotation_mode = 'XYZ'
		cam_ob.rotation_euler = (1.1, 0, -1.57)
		bpy.data.cameras['Camera'].lens = 20

		bpy.context.scene.camera = self.scene.objects["Camera"]
		
		if bpy.data.objects.get("Observer") is None:
			mesh = bpy.data.meshes.new("Observer")
			bm = bmesh.new()
			bm.verts.new(cam_ob.location)
			bm.to_mesh(mesh)
			observer = bpy.data.objects.new("Observer", mesh)    
			bpy.context.collection.objects.link(observer)
			#self.scene.objects.link(observer)
			bm.free()
			#self.scene.update()
		else: 
			observer = bpy.data.objects["Observer"]            

		dg = bpy.context.evaluated_depsgraph_get() 
		dg.update()

		observer_entity = Entity(observer)
		observer_entity.camera = cam_ob
		observer_entity.location = np.array(cam_ob.location)
		observer_entity.up = np.array([0, 1, 3])
		observer_entity.right = np.array([1, 0, 0])
		observer_entity.set_frontal(observer_entity.location)
		return observer_entity

	def create_block(self, name="", location=(0,0,0), rotation=(0,0,0), material=None):
		if bpy.data.objects.get(name) is not None:
			bl = bpy.data.objects[name]
			bl.rotation_euler = rotation
			return bl
		block_mesh = bpy.data.meshes.new('Block_mesh')
		block = bpy.data.objects.new(name, block_mesh)
		bpy.context.collection.objects.link(block)

		bm = bmesh.new()
		bmesh.ops.create_cube(bm, size=self.block_edge)
		bm.to_mesh(block_mesh)
		bm.free()
		block.data.materials.append(material)
		block.location = location
		block.rotation_euler = rotation
		block['id'] = "bw.item.block." + name
		block['color_mod'] = material.name
		block['main'] = 1.0
		bpy.context.evaluated_depsgraph_get().update()
		return block

	def scene_setup(self):        
		bpy.data.materials.new(name="red")
		bpy.data.materials.new(name="blue")
		bpy.data.materials.new(name="green")
		bpy.data.materials['red'].diffuse_color = (1, 0, 0, 0)
		bpy.data.materials['green'].diffuse_color = (0, 1, 0, 0)
		bpy.data.materials['blue'].diffuse_color = (0, 0, 1, 0)

		self.block_names = ['Target', 'Starbucks', 'Twitter', 'Texaco', 'McDonald\'s', 'Mercedes', 'Toyota', 'Burger King']
		materials = [bpy.data.materials['blue'], bpy.data.materials['green'], bpy.data.materials['red']]
	
		self.blocks = [self.create_block(name, Vector((0, 0, self.block_edge / 2)), (0,0,0), materials[self.block_names.index(name) % 3]) for name in self.block_names]
		self.block_by_ids = {}
		self.block_to_ids = {}
		dg = bpy.context.evaluated_depsgraph_get().update()        

	def clear_scene(self):
		"""
		Remove every mesh from the scene
		"""
		
		#iterate over the objects in the scene
		for ob in bpy.data.objects:
			#If it's a mesh select it
			if ob.type == "MESH":
				ob.select = True
		#remove all selected objects
		bpy.ops.object.delete()
   
	def get_dimensions(self):
		"""
		Compute the dimensions of the salient part of the world
		by finding the smallest bounding box containing all the 
		objects.
		"""
		x_min = [entity.x_min for entity in self.entities]
		x_max = [entity.x_max for entity in self.entities]
		y_min = [entity.y_min for entity in self.entities]
		y_max = [entity.y_max for entity in self.entities]
		z_min = [entity.z_min for entity in self.entities]
		z_max = [entity.z_max for entity in self.entities]

		return [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

	def show_bbox(self, entity):
		"""Displays the bounding box around the entity in the scene."""
		mesh = bpy.data.meshes.new(entity.name + '_mesh')
		obj = bpy.data.objects.new(entity.name + '_bbox', mesh)
		self.scene.objects.link(obj)
		self.scene.objects.active = obj
		bbox = entity.bbox
		mesh.from_pydata(bbox, [], [(0, 1, 3, 2), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5), (4, 5, 7, 6)])
		mesh.update()

	def unoccluded(self, block):
		LeftBlocked = False
		RightBlocked = False
		for key in self.block_dict:
			if self.block_dict[key] != block:
				dist_left = get_distance_from_line(block.location, self.kinectLeft, self.block_dict[key].location)
				dist_right = get_distance_from_line(block.location, self.kinectRight, self.block_dict[key].location)
				if dist_left <= 0.05:
					LeftBlocked = True
				if dist_right <= 0.05:
					RightBlocked = True
		return LeftBlocked and RightBlocked

	def init_event_log(self):
		if not os.path.exists("logs"):
			os.makedirs("logs")
		with open("logs" + os.sep + "event_log", 'a+') as log:
			log.write("\nSESSION: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
			log.write("================================================================================\n")	

	def log_event(self, event, content):
		with open("logs" + os.sep + "event_log", 'a+') as log:
			elapsed =  time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))
			log.write('(' + elapsed + ') ' + event + ': ' + content + '\n')

	def update(self, block_data):
		moved_blocks = []
		updated_blocks = {}
		unpaired = []
		unpaired_blocks = [bl for bl in self.blocks]

		for block in self.blocks:
			updated_blocks[block] = 0



		for id, location, rotation in block_data:
			if id in self.block_by_ids:
				block = self.block_by_ids[id]              
				rot1 = np.array([item for item in rotation])
				rot2 = np.array([item for item in block.rotation_euler])				
				#Threshold changed from 0.05
				if np.linalg.norm(location - block.location) >= 0.1: # or np.linalg.norm(rot1 - rot2) >= 0.4: #0.05
					if self.verbose or self.verbose_rotation:
						if np.linalg.norm(location - block.location) >= 0.1:
							print ("MOVED BLOCK: ", block.name, location, block.location, np.linalg.norm(location - block.location))
						else:
							print ("ROTATED BLOCK: ", block.name, rotation, block.rotation_euler)
					moved_blocks.append(block.name)
					block.location = location
					#block.rotation_euler = rotation
				updated_blocks[block] = 1
			# else:
			# 	unpaired.append((id, location, rotation))
			else:
				id_assigned = False
				for block in self.blocks:
					if np.linalg.norm(location - block.location) < 0.1:
						if self.verbose:
							print ("NOISE: ", block.name, location, block.location, np.linalg.norm(location - block.location))
						self.block_by_ids.pop(self.block_to_ids[block], None)
						self.block_by_ids[id] = block
						self.block_to_ids[block] = id
						block.location = location
						#block.rotation_euler = rotation
						id_assigned = True
						updated_blocks[block] = 1
						#moved_blocks.append(block.name)
						break
				if id_assigned == False:
					unpaired.append((id, location, rotation))

		for id, location, rotation in unpaired:
			min_dist = 10e9
			cand = None
			for block in self.blocks:
				if updated_blocks[block] == 0:
					cur_dist = np.linalg.norm(location - block.location)
					if min_dist > cur_dist:
						min_dist = cur_dist
						cand = block
			if cand != None:
				if self.verbose or self.verbose_rotation:
					if np.linalg.norm(location - cand.location) >= 0.1:
						print ("MOVED BLOCK: ", cand.name, location, cand.location, np.linalg.norm(location - cand.location))                
					else:
						print ("ROTATED BLOCK: ", block.name, rotation, block.rotation_euler)
				self.block_by_ids.pop(self.block_to_ids[cand], None)
				self.block_by_ids[id] = cand
				self.block_to_ids[cand] = id
				updated_blocks[cand] = 1
				if np.linalg.norm(location - cand.location) >= 0.3:# or np.linalg.norm(rot1 - rot2) >= 0.4:
					cand.location = location
					#cand.rotation_euler = rotation
					moved_blocks.append(cand.name)
		return moved_blocks

	def update_state(self):
		block_data = self.get_block_data()		
		moved_blocks = self.update(block_data)

		print ("MOVED BLOCKS: ", moved_blocks)
		
		if len(self.history) == 0:
			moved_blocks = [ent.name for ent in self.entities if 'block' in ent.type_structure]

		for ent in self.entities:
			ent.update()
		# for name in moved_blocks:
		# 	ent = self.find_entity_by_name(name)
		# 	# old_loc = ent.location                    
		# 	# self.entities.remove(ent)
			
		# 	# ent = Entity(bpy.data.objects[name])
		# 	# self.entities.append(ent)
		# 	ent.update()
			
		
		# 	#print ("ENTITY {} IS RELOCATED BY DIST {}; OLD LOC: {}; NEW LOC: {}".format(ent.name, np.linalg.norm(old_loc - ent.location), old_loc, ent.location))
		# 	if self.verbose:
		# 		print ("ENTITY RELOCATED: {} {}; OLD LOC: {}; NEW LOC: {}".format(ent.name, np.linalg.norm(old_loc - ent.location), old_loc, ent.location))
		bpy.context.evaluated_depsgraph_get().update()

		if len(moved_blocks) > 0:
			# self.history.append(self.State(self.entities))
			self.record_history()
			if len(self.history) == 1:
				self.make_checkpoint()

	def make_checkpoint(self):
		#self.checkpoint = self.history[-1]
		self.checkpoint = len(self.history)-1

	def move_to_ulf(self, name, loc1, loc2):
		def loc_to_str(loc):
			return ' '.join([str(coord) for coord in loc])
		
		return '(|' + name + '| ((past move.v) (from.p-arg ($ loc ' + loc_to_str(loc1) + ')) (to.p-arg ($ loc ' +\
							loc_to_str(loc2) + '))))'

	def record_history(self):
		self.history.append(self.State(self.entities))
		if len(self.history) > 1:
			moves = self.history[-1].state_diff(self.history[-2])
			for move in moves:
				self.log_event('BLOCK MOVE', self.move_to_ulf(move[0], move[1], move[2]))
			print (self.history[-1].state_diff(self.history[-2]))

	def get_last_move(self):
		return moves[-1]

	def get_moves_after_checkpoint(self):
		result = []
		for i in range(self.checkpoint+1, len(self.history)):
			result += self.history[i].state_diff(self.history[i-1])
		#return self.history[-1].state_diff(self.checkpoint)
		return result

	# def get_last_moved(self):
	# 	if self.history == []:
	# 		return None
	# 	elif len(self.history) == 1:
	# 		return [[key, self.history[0].locations[key]] for key in self.history[0].locations]
	# 	else:
	# 		ret_val = []
	# 		for key in self.history[-1].locations:
	# 			if key not in self.history[-2].locations or \
	# 					np.linalg.norm(self.history[-2].locations[key] - self.history[-1].locations[key]) > 0.1:					
	# 				ret_val.append([key, self.history[-1].locations[key]])
	# 		return ret_val

	def find_entity_by_name(self, name):
		"""
		Search and return the entity that has the given name
		associated with it.

		Inputs: name - human-readable name as a string

		Returns: entity (if exists) or None.
		"""

		for entity in self.entities:
			if entity.name.lower() == name.lower():
				return entity
		

		for col in self.color_mods:
			if col in name:
				name = name.replace(col + " ", "")				
		for entity in self.entities:			
			if entity.name.lower() == name.lower():
				return entity
		return None

	class ModalTimerOp(bpy.types.Operator):
		#metatags for Blender internal machinery
		bl_idname = "wm.modal_timer_operator"
		bl_label = "Modal Timer Op"
		
		#internal timer
		_timer = None
		world = None       
		
		#execution step (fires at every timer tick)
		def modal(self, context, event):
			if event.type == "ESC":
				return self.cancel(context)
			elif event.type == "TIMER":
				self.world.update_state()
									
			return {"PASS_THROUGH"}
		
		#Setup code (fires at the start)
		def execute(self, context):
			self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
			context.window_manager.modal_handler_add(self)
			return {"RUNNING_MODAL"}
		
		#Timer termination and cleanup
		def cancel(self, context):
			context.window_manager.event_timer_remove(self._timer)
			return {"CANCELLED"}

	class ProcessInputOp(bpy.types.Operator):
		"""Process input while Control key is pressed."""
		#bl_idname = 'view3d.process_input'
		bl_idname = 'view3d.process_input'
		bl_label = 'ProcessInput'
		bl_options = {'REGISTER'}
		world = None
		move_mode = False
		actual_move = False

		def modal(self, context, event):			
			# if event.type == 'ESC':			
			# 	return {'FINISHED'}						
			if event.ctrl and event.shift and event.type == 'M':
				def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):
					def draw(self, context):
						self.layout.label(text=message)

					bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

				if self.move_mode == False:
					self.move_mode = True
					ShowMessageBox('Move begins...')
				elif self.actual_move:
					for ent in self.world.entities:
						ent.update()
					self.world.record_history()					
					ShowMessageBox('Move complete...')
					self.actual_move = False
					self.move_mode = False					
			elif event.type == 'G' and self.move_mode:
				self.actual_move = True

			# elif event.type == 'G':
			# 	print("G pressed...")
			# elif event.type == 'LEFTMOUSE':
			# 	for ent in self.world.entities:
			# 		ent.update()
			# 	self.world.history.append(self.world.State(self.world.entities))
			# 	if len(self.world.history) > 1:
			# 		print (self.world.history[-1].state_diff(self.world.history[-2]))						
			return {'PASS_THROUGH'}

		def execute(self, context):
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}

		# def invoke(self, context, event):
		# 	print ("MODAL2")
		# 	context.window_manager.modal_handler_add(self)
		# 	return {'RUNNING_MODAL'}

	class State:
		def __init__(self, entities):
			self.locations = {}
			for ent in entities:
				self.locations[ent.name] = np.round(ent.location, 3)
			
		def state_diff(self, s):
			"""self - s"""
			result = []
			for name in self.locations:
				if np.linalg.norm(self.locations[name] - s.locations[name]) > 0:					
					print ("MOVE DIST: {}, {}".format(name, np.linalg.norm(self.locations[name] - s.locations[name])))
				if name in s.locations and np.linalg.norm(self.locations[name] - s.locations[name]) > 0.3:					
					result.append([name, self.locations[name], s.locations[name]])
			return result

		def compute(self):
			from constraint_solver import func_to_rel_map
			relations = [spatial.to_the_left_of_deic, spatial.to_the_right_of_deic, spatial.near, spatial.at, spatial.between, spatial.on, spatial.in_front_of_deic]
			for ent1 in self.entities:
				for ent2 in self.entities:
					if ent1 != ent2:
						for rel in relations:
							if rel != spatial.between:
								val = rel(ent1, ent2)
								if val > 0.7:
									self.state_facts.append([func_to_rel_map[rel], ent1, ent2, val])