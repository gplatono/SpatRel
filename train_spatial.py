import os
import sys
from spatial2 import Spatial

spatial = None

path_to_data = sys.argv[1]

scene_path = path_to_data + os.sep + "scenes"
ann_path = path_to_data + os.sep + "annotations"

scenes = glob.glob(scene_path + os.sep + "*.blend")
annotations = glob.glob(scene_path + os.sep + "*.data")

print (scenes, annotations)

def train(epochs):
	scene_idx = 0
	for epoch in range(epochs):
		scene = scenes[scene_idx]		
	    print(scene, scene.split('.')[0] + '.data')
	    if scene.split('.')[0] + '.data' in tests:
	        command = ['blender', scene, '-P', 'test_scene.py', '--', scene.split('.')[0] + '.data']
	        subprocess.run(command)