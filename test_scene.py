import bpy
import sys
import os
import bmesh
import numpy as np
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

import spatial2
from entity import Entity

scene = bpy.context.scene
entities = []
observer = None

right = spatial2.RightOf()
left = spatial2.LeftOf()

sp_map = {
	'to the left of' : left,
	'to the right of': right
}


def get_observer():
	global observer
	if observer == None:
		observer = create_observer()

def create_observer():
	"""Create and configure the special "observer" object
	(which is just a camera). Needed for deictic relations as
	well as several other aspects requiring the POV concept,
	e.g., taking screenshots.
	"""
	
	#lamp = bpy.data.lamps.new("Lamp", type = 'POINT')
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
		
	lamp_obj.location = (0, -20, 10)
	cam_ob.location = (0, -9, 3)
	cam_ob.rotation_mode = 'XYZ'
	cam_ob.rotation_euler = (1.1, 0, -1.57)
	bpy.data.cameras['Camera'].lens = 20

	bpy.context.scene.camera = scene.objects["Camera"]
	
	if bpy.data.objects.get("Observer") is None:
		mesh = bpy.data.meshes.new("Observer")
		bm = bmesh.new()
		bm.verts.new(cam_ob.location)
		bm.to_mesh(mesh)
		observer = bpy.data.objects.new("Observer", mesh)    
		bpy.context.collection.objects.link(observer)
		bm.free()		
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

def find_by_name(name):
	"""
	Search and return the entity that has the given name
	associated with it.

	Inputs: name - human-readable name as a string

	Returns: entity (if exists) or None.
	"""

	for entity in entities:
		if entity.name.lower() == name.lower():
			return entity
			
	return None

def run_testcase(testcase):
	components = testcase.split(':')
	print (components)
	subj = find_by_name(components[0])
	obj1 = find_by_name(components[2])
	obj2 = find_by_name(components[3]) if len(components) == 4 else None
	if components[1] not in sp_map:
		return -1
	rel = sp_map[components[1]]
	if obj2 is None:
		return rel.compute(subj, obj1)
	else:
		return rel.compute(subj, obj1, obj2)

test_file = sys.argv[-1]
spatial2.observer = create_observer()
# for obj in bpy.data.objects:
# 	print (obj.name)

for obj in scene.objects:
	if obj.get('main') is not None and obj.get('enabled') is None:
		entities.append(Entity(obj))	

tests = []

with open(test_file) as f:
 	tests = [line.strip() for line in f.readlines()]

print (tests)

for test in tests:
	print (run_testcase(test))