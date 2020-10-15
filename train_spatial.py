import os
import sys
import glob
import subprocess
import json

path_to_data = sys.argv[1]

scene_path = path_to_data + os.sep + "scenes"
ann_path = path_to_data + os.sep + "annotations"

scenes = glob.glob(scene_path + os.sep + "*.blend")
annotations = glob.glob(ann_path + os.sep + "*.data")

#print (scenes, annotations)

def train(epochs):
	scene_idx = 0
	rel_acc = None
	actual_epoch = 0
	for epoch in range(epochs):
		scene = scenes[scene_idx]
		name = scene.split(os.sep)[-1].split(".blend")[0] + '.data'
		#print("SCENE DATA:", scene, name)
		if ann_path + os.sep + name in annotations and 'Jon' not in scene:
			#command = ['/Applications/Blender.app/Contents/MacOS/Blender', scene, '-P', 'train_scene.py', '--', ann_path + os.sep + name]
			command = ['../blender/blender', scene, '--background', '-P', 'train_scene.py', '--', ann_path + os.sep + name]
			subprocess.run(command)
			
			tmp_acc = None
			with open('rel_accuracies', 'r') as file:
				tmp_acc = json.load(file)

			if tmp_acc is not None and tmp_acc != "":
				if rel_acc is None:
					rel_acc = tmp_acc.copy()
				else:
					for key in tmp_acc:
						rel_acc[key]['total'] += tmp_acc[key]['total']
						rel_acc[key]['acc'] += tmp_acc[key]['acc']

				print ()
				for key in tmp_acc:
					print (key.upper() + ", {} annotations, accuracy: {:.3f}".format(tmp_acc[key]['total'], tmp_acc[key]['acc']))
				print ()
				
			actual_epoch += 1

		scene_idx += 1
		if scene_idx == len(scenes):
			scene_idx = 0

	for key in rel_acc:
		rel_acc[key]['acc'] = rel_acc[key]['acc'] / actual_epoch
		print (key.upper() + ", {} total ann, avg accuracy: {:.3f}".format(rel_acc[key]['total'], rel_acc[key]['acc']))

if __name__ == "__main__":
	train(100)