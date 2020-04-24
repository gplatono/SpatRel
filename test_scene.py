import bpy
import sys
import os
import bmesh
import numpy as np

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from world import World
import spatial2

world = World(bpy.context.scene, simulation_mode=True)
spatial2.world = world
spatial2.observer = world.get_observer()
spatial_module = spatial2.Spatial()

def run_testcase(testcase):
	components = testcase.split(':')
	print (components)
	relation = components[1].strip()
	trs = [world.find_entity_by_name(components[0].strip())]
	lms = [world.find_entity_by_name(item.strip()) for item in components[2:]]	
	if relation != 'on' and relation != 'next to' and relation != 'touching':
		return spatial_module.compute(relation, trs, lms)

test_file = sys.argv[-1]

tests = []

with open(test_file) as f:
	tests = [line.strip() for line in f.readlines()]
	for test in tests:
		print (run_testcase(test))