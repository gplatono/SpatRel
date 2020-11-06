import bpy
import sys
import os

# print (sys.executable)
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)
#sys.path.append('/home/georgiy/.local/lib/python3.7/site-packages/')

from world import World
from spatial2 import Spatial

datapath = sys.argv[-1]
fname = datapath.split(os.sep)[-1]

old_stdout = sys.stdout # backup current stdout
sys.stdout = open(os.devnull, "w")

world = World(bpy.context.scene, simulation_mode=True)
spatial = Spatial(world)

sys.stdout = old_stdout

objlist = [name.name for name in world.entities]
#print(objlist)

def test():
    with open(datapath, "r") as f:
        flag = 0
        lines = f.readlines()
        annotations = [line.split(":") for line in lines if line.strip() != ""]
        for i, ele in enumerate(annotations):
            if len(ele) != 3:
                if len(ele) != 4 or ele[1] !='between':
                    print("Colon separation error found in line " + str(i+1))
                    print(ele)
                    print(fname + '\n')
                    flag = 1
        if flag == 1:
            return
        print('colon number correct \n')

        for i, ele in enumerate(annotations):
            if len(ele) != 3:
                if len(ele) != 4 or ele[1] != 'between':
                    print("error")
            for idx in range(len(ele)):
                if idx != 1:
                    try:
                        trs = spatial.world.find_entity_by_name(ele[idx].strip())
                        trs = trs.name
                    except:
                        print('Following obj does not exist: ' + ele[idx])
                        print('Found in line ' + str(i+1))
                        print(fname + '\n')

    print('annotations check finished')

test()