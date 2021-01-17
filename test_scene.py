import bpy
import sys
import os
import bmesh
import numpy as np
import json

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from world import World
import spatial2

types_ids = {
	'chair': 'props.item.furniture.chair',
	'table': 'props.item.furniture.table',    
	'long table': 'props.item.furniture.table.long table',    
	'coffee table': 'props.item.furniture.table.coffee table',    
	'bed': 'props.item.furniture.bed',
	'sofa':  'props.item.furniture.sofa',
	'bookshelf':  'props.item.furniture.bookshelf',
	'desk':  'props.item.furniture.desk',
	'book': 'props.item.portable.book',
	'laptop': 'props.item.portable.laptop',
	'pencil': 'props.item.portable.pencil',
	'pencil holder': 'props.item.portable.pencil holder',
	'note': 'props.item.portable.note',
	'rose': 'props.item.portable.rose',
	'vase': 'props.item.portable.vase',
	'cardbox': 'props.item.portable.cardbox',
	'box': 'props.item.portable.box',
	'ceiling light': 'props.item.stationary.ceiling light',
	'lamp': 'props.item.portable.lamp',
	'floor lamp': 'props.item.portable.lamp.floor lamp',
	'apple': 'props.item.food.apple',
	'banana': 'props.item.food.banana',
	'plate': 'props.item.portable.plate',
	'bowl': 'props.item.portable.bowl',
	'trash bin': 'props.item.portable.trash can',
	'trash can': 'props.item.portable.trash can',
	'tv': 'props.item.appliances.tv',
	'poster': 'props.item.stationary.poster',
	'picture': 'props.item.stationary.picture',
	'fridge' : 'props.item.appliances.fridge',
	'ceiling fan': 'props.item.stationary.ceiling fan',
	'block': 'props.item.block',
	'floor': 'world.plane.floor',
	'ceiling': 'world.plane.ceiling',
	'wall': 'world.plane.wall',
	'block 5': 'item.prop.block.Block 5',
	'block 4': 'item.prop.block.Block 4',
	'block 3': 'item.prop.block.Block 3',
	'block 2': 'item.prop.block.Block 2',
	'block 1': 'item.prop.block.Block 1'
}

color_mods = ['black', 'red', 'blue', 'brown', 'green', 'yellow']


def fix_ids():
	for ob in bpy.context.scene.objects:
		if ob.get('main') is not None:
			cand = None
			for key in types_ids.keys():
				if key in ob.name.lower() and (cand is None or cand in key):
					cand = key

			ob['id'] = types_ids[cand] + "." + ob.name
			if ob.get('color_mod') is None:
				for color in color_mods:
					if color in ob.name.lower():
						ob['color_mod'] = color
						break

#fix_ids()
#bpy.ops.wm.save_mainfile(filepath=bpy.data.filepath)

world = World(bpy.context.scene, simulation_mode=True)
spatial2.world = world
spatial2.observer = world.get_observer()
spatial_module = spatial2.Spatial(world)

# print ('computing canopy...')
# ent = world.find_entity_by_name('')
# canopy = ent.compute_canopy()
# print ('Span: ', ent.span)
# print ('Canopy: ', canopy)

def fix(components):
	ret_val = []
	for item in components:
		item.strip()
		item = item.replace('in between', 'between')
		item = item.replace('east of', 'to the right of')
		ret_val.append(item)

	return ret_val


def run_testcase(testcase):
	if testcase == '':
		return
	components = fix(testcase.split(':'))
	print (components)
	relation = components[1]
	trs = [world.find_entity_by_name(components[0].strip())]
	lms = [world.find_entity_by_name(item.strip()) for item in components[2:]]

	print (trs, lms)
	tr_data = [item.get_features() for item in trs]
	lm_data = [item.get_features() for item in lms]
	rel_to_label = {'on': 14, 'to the left of': 1, 'left of': 1, 'to the right of': 2, 'right of': 2, 'above': 3,
			'below': 4, 'in front of': 5, 'behind': 6, 'over': 7, 'under': 8, 'underneath': 8, 'in': 9, 'inside': 9,
			'touching': 10, 'touch': 10, 'at': 11, 'next to': 11, 'between': 12, 'near': 13, 'on top of': 14, 'beside': 15, 'besides': 15}
	if 'not ' not in relation:
		label = rel_to_label[relation]
	else:
		label = -rel_to_label[relation.replace('not ', '')]

	# if label != 3 and label != -3 and label != 4 and label != -4:
	#  	return

	#print (tr_data)
	observer_loc = list(map(lambda x: float(x), world.get_observer().location))
	observer_front = list(map(lambda x: float(x), world.get_observer().front))
	data = {'arg0' : tr_data, 'arg1': lm_data, 'label': label, 'observer_loc': observer_loc, 'observer_front': observer_front}
	print (data)
	with open('dataset', 'a+') as file:
		file.write(json.dumps(data) + '\n')
	# if relation != 'on' and relation != 'next to' and relation != 'touching':
	# print (trs, lms)
	if relation != 'on' and None not in trs and None not in lms:
		return spatial_module.compute(relation, trs, lms)

test_file = sys.argv[-1]

tests = []

with open(test_file) as f:
	print (test_file)	
	tests = [line.strip() for line in f.readlines()]
	for test in tests:
		print (run_testcase(test))

sys.exit()

# input()
#bpy.ops.wm.quit_blender()