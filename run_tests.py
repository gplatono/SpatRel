import glob
import subprocess

scenes = glob.glob("*.blend")
tests = glob.glob("*.data")

for scene in scenes:
    print(scene, scene.split('.')[0] + '.data')
    if scene.split('.')[0] + '.data' in tests:
        command = ['blender', scene, '-P', 'test_scene.py', '--', scene.split('.')[0] + '.data']
        subprocess.run(command)
