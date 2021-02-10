import glob
import subprocess
import sys
import os

path_to_data = sys.argv[1]

scene_path = path_to_data + os.sep + "scenes"
ann_path = path_to_data + os.sep + "annotations"
 
scenes = glob.glob(scene_path + os.sep + "*.blend")
annotations = glob.glob(ann_path + os.sep + "*.data")
for scene in scenes:    
    # if scene != 'RW3.blend':
    # 	continue
    ann_file = ann_path + os.sep + scene.split(os.sep)[-1].split(".blend")[0] + '.data'
    print(scene, ann_file)
    if ann_file in annotations:
        command = ['../blender/blender', scene, '-b', '-P', 'test_scene.py', '--', ann_file]
        subprocess.run(command)
