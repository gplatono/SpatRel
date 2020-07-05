import glob
import subprocess
import sys

path = sys.argv[1]
scenes = glob.glob("*.blend")
tests = glob.glob("*.data")

for scene in scenes:    
    # if scene != 'RW3.blend':
    # 	continue
    print(scene, scene.split('.')[0] + '.data')
    if scene.split('.')[0] + '.data' in tests:
        command = ['blender', scene, '-P', 'test_scene.py', '--', scene.split('.')[0] + '.data']
        subprocess.run(command)
