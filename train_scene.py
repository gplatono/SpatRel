import bpy
import sys
import os

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from world import World
from spatial2 import Spatial

datapath = sys.argv[1]

world = World(bpy.context.scene, simulation_mode=True)
#spatial2.world = world
#spatial2.observer = world.get_observer()
spatial = spatial2.Spatial(world)

with open(datapath, "r") as f:
	annotations = [line.split(":") for line in f.readlines()]
	spatial.train(annotations, 10)
	spatial.save_parameters()