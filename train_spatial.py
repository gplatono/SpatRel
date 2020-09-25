import os
import sys
import glob
import subprocess

path_to_data = sys.argv[1]

scene_path = path_to_data + os.sep + "scenes"
ann_path = path_to_data + os.sep + "annotations"

scenes = glob.glob(scene_path + os.sep + "*.blend")
annotations = glob.glob(ann_path + os.sep + "*.data")

#print (scenes, annotations)

def train(epochs):
	scene_idx = 0
	for epoch in range(epochs):
		scene = scenes[scene_idx]
		name = scene.split(os.sep)[-1].split(".blend")[0] + '.data'
		#print("SCENE DATA:", scene, name)
		if ann_path + os.sep + name in annotations and 'Jon' not in scene and name == "RW4.data":
			command = ['/Applications/Blender.app/Contents/MacOS/Blender', scene, '-P', 'train_scene.py', '--', ann_path + os.sep + name]
			# command = ['../blender-2.83.5-linux64/blender', scene, '-P', 'train_scene.py', '--', ann_path + os.sep + name]
			subprocess.run(command)
		scene_idx += 1
		if scene_idx == len(scenes):
			scene_idx = 0

if __name__ == "__main__":
	train(100)