import bpy
import sys
import os

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)
#sys.path.append('/home/georgiy/.local/lib/python3.7/site-packages/')

from world import World
from spatial2 import Spatial

datapath = sys.argv[-1]

world = World(bpy.context.scene, simulation_mode=True)
spatial = Spatial(world)


with open(datapath, "r") as f:
	lines = f.readlines()
	annotations = [line.split(":") for line in lines if line.strip() != ""]
	#spatial.reload(world)
	spatial.train(annotations, 10)
	spatial.save_parameters()
	bpy.ops.wm.quit_blender()

# for entity in world.entities:
# 	en_info = spatial.preprocessig(entity)
# 	print(en_info)