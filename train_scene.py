import bpy
import sys
import os

print (sys.executable)
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)
sys.path.append('/home/georgiy/.local/lib/python3.7/site-packages/')

from world import World
from spatial2 import Spatial

datapath = sys.argv[5]

world = World(bpy.context.scene, simulation_mode=True)
#spatial2.world = world
#spatial2.observer = world.get_observer()
spatial = Spatial(world)

with open(datapath, "r") as f:
	#print ("DATAPATH: ", datapath)
	lines = f.readlines()
	annotations = []
	for line in lines:
		if line.strip() != "":
			annotations.append(line)
	#annotations = [line.split(":") for line in f.readlines()]
	spatial.reload(world)
	spatial.train(annotations, 10)
	spatial.save_parameters()