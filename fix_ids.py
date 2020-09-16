import sys
import os
import numpy as np
import json
import subprocess
import sys
import glob

#reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
#installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

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
	import bpy

	for ob in bpy.context.scene.objects:
		if ob.get('main') is not None:
			cand = None
			for key in types_ids.keys():
				if key in ob.name.lower() and (cand is None or cand in key):
					cand = key
			
			if cand is not None:
				ob['id'] = types_ids[cand] + "." + ob.name
				if ob.get('color_mod') is None:
					for color in color_mods:
						if color in ob.name.lower():
							ob['color_mod'] = color
							break

def fix_n_save():
	import bpy
	
	fix_ids()
	bpy.ops.wm.save_mainfile(filepath=bpy.data.filepath)

def fix_scenes():
	scenes = glob.glob(sys.argv[1] + os.sep + "*.blend")
	for scene in scenes:
		subprocess.run(['../blender-2.83.5-linux64/blender', scene, '--background', '-P', 'fix_ids.py'])

if __name__ == "__main__":
	if len(sys.argv) <= 2:		
		fix_scenes()
	else:
		fix_n_save()