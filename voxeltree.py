import numpy as np
import bmesh
from mathutils.bvhtree import BVHTree
from random import sample
import os
import sys
import time

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)

from geometry_utils import *
from world import World


class Voxel:
    def __init__(self, scope, location=None, size=None, parent=None, depth=0, child_idx=[None, None, None]):
        if location is None:
            x_min = y_min = z_min = 1e9
            x_max = y_max = z_max = -1e9
            for entity in world.entities:
                x_min = min(x_min, entity.x_min)
                y_min = min(y_min, entity.y_min)
                z_min = min(z_min, entity.z_min)
                x_max = max(x_max, entity.x_max)
                y_max = max(y_max, entity.y_max)
                z_max = max(z_max, entity.z_max)
            location = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)])
            size = max([x_max - x_min, y_max - y_min, z_max - z_min])

        self.location = location
        self.size = size
        self.depth = depth
        self.child_idx = child_idx
        self.compute_bbox()
        self.intersect_list = self.filter_entity(scope)
        self.parent = parent
        self.children = [[[None, None], [None, None]], [[None, None], [None, None]]]
        self.neighbors = [[None, None], [None, None], [None, None]]

        self.subdivide()
        self.node_count = self.get_node_count()

    # self.fillNeighbors()

    def run_on_children(self, method_name):
        ret_val = []
        if self.children[0][0][0] is not None:
            ret_val += [getattr(self.children[0][0][0], method_name)(),
                        getattr(self.children[0][0][1], method_name)(),
                        getattr(self.children[0][1][0], method_name)(),
                        getattr(self.children[0][1][1], method_name)(),
                        getattr(self.children[1][0][0], method_name)(),
                        getattr(self.children[1][0][1], method_name)(),
                        getattr(self.children[1][1][0], method_name)(),
                        getattr(self.children[1][1][1], method_name)()]

        return ret_val

    def get_node_count(self):
        return 1 + sum(self.run_on_children("get_node_count"))

    # if self.children[0][0][0] is not None:
    # 	ret_val += self.children[0][0][0].get_size()
    # 	ret_val += self.children[0][0][1].get_size()
    # 	ret_val += self.children[0][1][0].get_size()
    # 	ret_val += self.children[0][1][1].get_size()
    # 	ret_val += self.children[1][0][0].get_size()
    # 	ret_val += self.children[1][0][1].get_size()
    # 	ret_val += self.children[1][1][0].get_size()
    # 	ret_val += self.children[1][1][1].get_size()

    # return ret_val

    def print_self(self):
        print("\n", self, self.child_idx, self.location, self.size, self.parent, self.node_count)
        print(self.children)
        print(self.neighbors)

    def printStructure(self):
        self.print_self()
        self.run_on_children("printStructure")

    def subdivide(self):
        quart = self.size / 4
        if self.depth > 0 and self.intersect_list != []:
            child_locations = np.array([(self.location[0] - quart, self.location[1] - quart, self.location[2] - quart),
                                        (self.location[0] - quart, self.location[1] - quart, self.location[2] + quart),
                                        (self.location[0] - quart, self.location[1] + quart, self.location[2] - quart),
                                        (self.location[0] - quart, self.location[1] + quart, self.location[2] + quart),
                                        (self.location[0] + quart, self.location[1] - quart, self.location[2] - quart),
                                        (self.location[0] + quart, self.location[1] - quart, self.location[2] + quart),
                                        (self.location[0] + quart, self.location[1] + quart, self.location[2] - quart),
                                        (self.location[0] + quart, self.location[1] + quart, self.location[2] + quart)])
            self.children[0][0][0] = Voxel(self.intersect_list, child_locations[0], self.size / 2, self, self.depth - 1,
                                           [0, 0, 0])
            self.children[0][0][1] = Voxel(self.intersect_list, child_locations[1], self.size / 2, self, self.depth - 1,
                                           [0, 0, 1])
            self.children[0][1][0] = Voxel(self.intersect_list, child_locations[2], self.size / 2, self, self.depth - 1,
                                           [0, 1, 0])
            self.children[0][1][1] = Voxel(self.intersect_list, child_locations[3], self.size / 2, self, self.depth - 1,
                                           [0, 1, 1])
            self.children[1][0][0] = Voxel(self.intersect_list, child_locations[4], self.size / 2, self, self.depth - 1,
                                           [1, 0, 0])
            self.children[1][0][1] = Voxel(self.intersect_list, child_locations[5], self.size / 2, self, self.depth - 1,
                                           [1, 0, 1])
            self.children[1][1][0] = Voxel(self.intersect_list, child_locations[6], self.size / 2, self, self.depth - 1,
                                           [1, 1, 0])
            self.children[1][1][1] = Voxel(self.intersect_list, child_locations[7], self.size / 2, self, self.depth - 1,
                                           [1, 1, 1])

    def compute_bbox(self):
        self.bbox_verts = np.array(
            [(self.location[0] - self.size / 2, self.location[1] - self.size / 2, self.location[2] - self.size / 2),
             (self.location[0] - self.size / 2, self.location[1] - self.size / 2, self.location[2] + self.size / 2),
             (self.location[0] - self.size / 2, self.location[1] + self.size / 2, self.location[2] - self.size / 2),
             (self.location[0] - self.size / 2, self.location[1] + self.size / 2, self.location[2] + self.size / 2),
             (self.location[0] + self.size / 2, self.location[1] - self.size / 2, self.location[2] - self.size / 2),
             (self.location[0] + self.size / 2, self.location[1] - self.size / 2, self.location[2] + self.size / 2),
             (self.location[0] + self.size / 2, self.location[1] + self.size / 2, self.location[2] - self.size / 2),
             (self.location[0] + self.size / 2, self.location[1] + self.size / 2, self.location[2] + self.size / 2)])
        self.bbox_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [2, 3, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]]
        self.bvh_tree = BVHTree.FromPolygons(self.bbox_verts, self.bbox_faces, epsilon=1.0)

    def findContainingVoxel(self, point, depth=-1):
        child_x = 0 if point[0] <= self.location[0] else 1
        child_y = 0 if point[1] <= self.location[1] else 1
        child_z = 0 if point[2] <= self.location[2] else 1

        if depth == 0 or self.children[child_x][child_y][child_z] is None:
            return self if box_point_containment(self.bbox_verts, point) else None

        return self.children[child_x][child_y][child_z].findContainingVoxel(point, depth - 1)

    def fillNeighbors(self):
        if self.neighbors[0][0] is None:
            self.XDownAdjacent()
        if self.neighbors[0][1] is None:
            self.XUpAdjacent()
        if self.neighbors[1][0] is None:
            self.YDownAdjacent()
        if self.neighbors[1][1] is None:
            self.YUpAdjacent()
        if self.neighbors[2][0] is None:
            self.ZDownAdjacent()
        if self.neighbors[2][1] is None:
            self.ZUpAdjacent()

        if self.children[0][0][0] is not None:
            self.children[0][0][0].fillNeighbors()
            self.children[0][0][1].fillNeighbors()
            self.children[0][1][0].fillNeighbors()
            self.children[0][1][1].fillNeighbors()
            self.children[1][0][0].fillNeighbors()
            self.children[1][0][1].fillNeighbors()
            self.children[1][1][0].fillNeighbors()
            self.children[1][1][1].fillNeighbors()

    def XDownAdjacent(self):
        if self.child_idx[0] == 1:
            adjacent = self.parent.children[0][self.child_idx[1]][self.child_idx[2]]
        else:
            LCA = self
            while LCA.child_idx[0] == 0:
                LCA = LCA.parent
            adjacent = LCA.findContainingVoxel(self.location - np.array([self.size, 0, 0]))
        if adjacent is not None and adjacent.size == self.size:
            self.neighbors[0][0] = adjacent
            adjacent.neighbors[0][1] = self

    def XUpAdjacent(self):
        if self.child_idx[0] == 0:
            adjacent = self.parent.children[1][self.child_idx[1]][self.child_idx[2]]
        else:
            LCA = self
            while LCA.child_idx[0] == 1:
                LCA = LCA.parent
            adjacent = LCA.findContainingVoxel(self.location + np.array([self.size, 0, 0]))
        if adjacent is not None and adjacent.size == self.size:
            self.neighbors[0][1] = adjacent
            adjacent.neighbors[0][0] = self

    def YDownAdjacent(self):
        if self.child_idx[1] == 1:
            adjacent = self.parent.children[self.child_idx[0]][0][self.child_idx[2]]
        else:
            LCA = self
            while LCA.child_idx[1] == 0:
                LCA = LCA.parent
            adjacent = LCA.findContainingVoxel(self.location - np.array([0, self.size, 0]))
        if adjacent is not None and adjacent.size == self.size:
            self.neighbors[1][0] = adjacent
            adjacent.neighbors[1][1] = self

    def YUpAdjacent(self):
        if self.child_idx[1] == 0:
            adjacent = self.parent.children[self.child_idx[0]][1][self.child_idx[2]]
        else:
            LCA = self
            while LCA.child_idx[1] == 1:
                LCA = LCA.parent
            adjacent = LCA.findContainingVoxel(self.location + np.array([0, self.size, 0]))
        if adjacent is not None and adjacent.size == self.size:
            self.neighbors[1][1] = adjacent
            adjacent.neighbors[1][0] = self

    def ZDownAdjacent(self):
        if self.child_idx[2] == 1:
            adjacent = self.parent.children[self.child_idx[0]][self.child_idx[1]][0]
        else:
            LCA = self
            while LCA.child_idx[2] == 0:
                LCA = LCA.parent
            adjacent = LCA.findContainingVoxel(self.location - np.array([0, 0, self.size]))
        if adjacent is not None and adjacent.size == self.size:
            self.neighbors[2][0] = adjacent
            adjacent.neighbors[2][1] = self

    def ZUpAdjacent(self):
        if self.child_idx[2] == 0:
            adjacent = self.parent.children[self.child_idx[0]][self.child_idx[1]][1]
        else:
            LCA = self
            while LCA.child_idx[2] == 1:
                LCA = LCA.parent
            adjacent = LCA.findContainingVoxel(self.location + np.array([0, 0, self.size]))
        if adjacent is not None and adjacent.size == self.size:
            self.neighbors[2][1] = adjacent
            adjacent.neighbors[2][0] = self


    def contains(self, entities, depth=-1):
        flag = False
        for ent in entities:
            for ele in self.intersect_list:
                if ent == ele[0]:
                    flag = True
            if flag == False:
                return False
            flag = False

        if depth == 0 or self.children[0][0][0] is None:
            return True

        child_res = False
        child_res = child_res or self.children[0][0][0].contains(entities, depth - 1)
        child_res = child_res or self.children[0][0][1].contains(entities, depth - 1)
        child_res = child_res or self.children[0][1][0].contains(entities, depth - 1)
        child_res = child_res or self.children[0][1][1].contains(entities, depth - 1)
        child_res = child_res or self.children[1][0][0].contains(entities, depth - 1)
        child_res = child_res or self.children[1][0][1].contains(entities, depth - 1)
        child_res = child_res or self.children[1][1][0].contains(entities, depth - 1)
        child_res = child_res or self.children[1][1][1].contains(entities, depth - 1)

        return child_res

    def filter_entity(self, scope):
        output = []
        for ent in scope:
            vertex = box_vertex_filter(self.bbox_verts, ent[1])
            faces = []
            for f in ent[2]:
                if len(f) == 3:
                    if box_point_containment(self.bbox_verts, f[0]) or box_point_containment(self.bbox_verts, f[1]) or box_point_containment(self.bbox_verts, f[2]):
                        faces.append(f)
                    else:
                        closest_point = self.location + shortest_to_triangle(self.location, f[0], f[1], f[2]).tolist()
                        if box_point_containment(self.bbox_verts, closest_point):
                            faces.append(f)
                elif len(f) == 4:
                    if box_point_containment(self.bbox_verts, f[0]) or box_point_containment(self.bbox_verts, f[1]) or \
                            box_point_containment(self.bbox_verts, f[2]) or box_point_containment(self.bbox_verts, f[3]):
                        faces.append(f)
                    else:
                        closest_point1 = self.location + shortest_to_triangle(self.location, f[0], f[1], f[2]).tolist()
                        closest_point2 = self.location + shortest_to_triangle(self.location, f[3], f[1], f[2]).tolist()
                        if box_point_containment(self.bbox_verts, closest_point1) or box_point_containment(self.bbox_verts, closest_point2):
                            faces.append(f)
            if not (len(vertex) == 0 and len(faces) == 0):
                output.append([ent[0], vertex, faces])

        #if there are only faces in the box or there is only 1 entity in the box, the box is filtered out.
        count_vertex = 0
        for ent in output:
            if len(ent[1]) > 0:
                count_vertex += 1
        if count_vertex == 0 or len(output) < 2:
            return []

        return output



def entitymap(entities, c):
    output = []
    for entity in entities:
        vertex = entity.vertex_set
        if len(entity.faces) > c:
            faces = sample(entity.faces, c)
        else:
            faces = entity.faces
        output.append([entity, vertex, faces])
    return output

def test(vox):
    i = input('enter lm, tr, depth: ')
    while i != 'q':
        tr, lm, d = i.split(":")
        tr = world.find_entity_by_name(tr)
        lm = world.find_entity_by_name(lm)
        print(vox.contains([tr, lm], depth=int(d)))
        i = input('enter lm, tr, depth: ')
    print("test end")


start = time.time()
world = World(bpy.context.scene, simulation_mode=True)

vox = Voxel(scope=entitymap(world.entities, 50), depth=8)
end = time.time()
print(end - start)

test(vox)