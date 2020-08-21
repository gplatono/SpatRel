import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import json
from spatial2 import *
from geometry_utils import *
import numpy as np

num_classes = 15

world = None


class Sample:
    def __init__(self, val):
        self.val = val
        self.centroid = (val, val, val)
        self.bbox = 0


# For custom objects
# def build_custom_loader(batch_size, num_workers, is_shuffle):
#     samples = []
#     labels = []
#     for i in range(1000):
#         samples.append(Sample(i))
#         labels.append(i * i)
#     dataset = Dataset(samples, labels)
#     train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
#     test_loader = train_loader
#     return train_loader, test_loader

class CustomNet:

    def __init__(self):
        self.params = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)

    def compute(self, sample):
        result = torch.dot(torch.tensor(sample.centroid, dtype=torch.float32), self.params)
        return result

    def parameters(self):
        return [self.params]


class Rightof_Deictic:

    def __init__(self):
        self.params = torch.tensor([0.4, 0.6], dtype=torch.float32, requires_grad=True)

    def compute(self, tr, lm):
        horizontal_component = HorizontalDeicticComponent.compute(tr, lm)
        vertical_component = VerticalDeicticComponent.compute(tr, lm)
        hv_component = torch.tensor([horizontal_component, vertical_component],
                                    dtype=torch.float32)  # transferred tensor
        final_score = torch.dot(hv_component, self.params)
        return final_score

    def str(self):
        return 'to_the_right_of_deictic.p'


class LeftOf_Deictic:

    def compute(self, tr, lm):
        return Rightof_Deictic.compute(tr=lm, lm=tr)

    def str(self):
        return 'to_the_left_of_deictic.p'


def sigmoid(x, a, b):
    return a / (1 + math.e ** (- b * x)) if b * x > -100 else 0


class InFrontOf_Deictic:

    def __init__(self):
        self.params = {"observer_dist_factor_weight": torch.tensor(0.5, dtype=torch.float32, requires_grad=True),
                       "projection_factor_weight": torch.tensor(0.5, dtype=torch.float32, requires_grad=True),
                       "projection_factor_scale": torch.tensor(-0.5, dtype=torch.float32, requires_grad=True)}

    def compute(self, tr, lm):
        a_bbox = get_2d_bbox(vp_project(tr, world.get_observer()))
        b_bbox = get_2d_bbox(vp_project(lm, world.get_observer()))
        a_center = projection_bbox_center(a_bbox)
        b_center = projection_bbox_center(b_bbox)
        dist = np.linalg.norm(a_center - b_center)
        scaled_proj_dist = dist / (max(get_2d_size(a_bbox), get_2d_size(b_bbox)) + 0.001)
        a_dist = np.linalg.norm(tr.location - world.observer.location)
        b_dist = np.linalg.norm(lm.location - world.observer.location)

        observer_dist_factor = sigmoid(b_dist - a_dist, 1, 0.5)
        # transfer to tensors
        observer_dist_factor = torch.tensor(observer_dist_factor, dtype=torch.float32)
        scaled_proj_dist = torch.tensor(scaled_proj_dist, dtype=torch.float32)
        projection_factor = math.e ** (self.parameters["projection_factor_scale"] * scaled_proj_dist)

        return self.parameters["observer_dist_factor_weight"] * observer_dist_factor \
               + self.parameters["projection_factor_weight"] * projection_factor

    def str(self):
        return 'in_front_of.p'


class Behind_Deictic(Node):
    def compute(self, tr, lm):
        return InFrontOf_Deictic.compute(tr=lm, lm=tr)

    def str(self):
        return 'behind_deictic.p'


class WithinConeRegion:

    def __init__(self):
        self.params = torch.tensor([2], dtype=torch.float32, requires_grad=True)

    def compute(self, vect, direction, width):
        cos = direction.dot(vect) / (np.linalg.norm(direction) * np.linalg.norm(vect))
        angle = math.acos(cos)
        final_score = 1 / (1 + math.e ** (
                self.param * torch.tensor([width - angle], dtype=torch.float32)))  # transfer to tensor
        return final_score


class Above:

    def compute(self, tr, lm):
        vertical_dist_scaled = (tr.centroid[2] - lm.centroid[2]) / (max(tr.dimensions[2], lm.dimensions[2]) + 0.01)
        return WithinConeRegion.compute(tr.centroid - lm.centroid, np.array([0, 0, 1.0]), 0.1) \
               * sigmoid(vertical_dist_scaled, 1, 3)

    def str(self):
        return 'above.p'


class Below(Node):

    def compute(self, tr, lm):
        return Above.compute(lm, tr)

    def str(self):
        return 'below.p'


class Touching:

    def __init__(self):
        self.parameters = {"touch_face_threshold": torch.tensor([0.95], dtype=torch.float32, requires_grad=True),
                           "mesh_dist_threshold": torch.tensor([0.1], dtype=torch.float32, requires_grad=True)}

    def compute(self, tr, lm):
        if tr == lm:
            return 0
        mesh_dist = 1e9
        planar_dist = 1e9
        shared_volume = torch.tensor([shared_volume_scaled(tr, lm)], dtype=torch.float32)

        if lm.get("planar") is not None:
            planar_dist = get_planar_distance_scaled(lm, tr)
        elif tr.get("planar") is not None:
            planar_dist = get_planar_distance_scaled(tr, lm)
        if get_centroid_distance_scaled(tr, lm) <= 1.5:
            mesh_dist = closest_mesh_distance(tr, lm) / (min(tr.size, lm.size) + 0.01)
        mesh_dist = torch.tensor([min(mesh_dist, planar_dist)], dytpe=torch.float32)
        touch_face = torch.tensor([0], dtype=torch.float32)
        if shared_volume == 0:
            if touch_face > self.parameters['touch_face_threshold']:
                ret_val = touch_face
            elif mesh_dist < self.parameters['mesh_dist_threshold']:
                ret_val = math.exp(- mesh_dist)
            else:
                ret_val = math.exp(- 2 * mesh_dist)
        else:
            ret_val = 0.3 * math.exp(- 2 * mesh_dist) + 0.7 * (shared_volume > 0)
        return ret_val


class Supported:

    def __init__(self):
        self.params = torch.tensor([0.8], dtype=torch.float32, requires_grad=True)

    def compute(self, tr, lm):
        direct_support = self.connections['touching'](tr, lm) * self.connections['above'](tr, lm)  # tensor
        indirect_support = torch.tensor([0], dtype=torch.float32)
        tr_h = torch.tensor([tr.centroid[2]], dtype=torch.float32)
        lm_h = torch.tensor([lm.centroid[2]], dtype=torch.float32)
        for entity in world.entities:
            e_h = torch.tensor([entity.centroid[2]], dtype=torch.float32)
            if (e_h - lm_h) / torch.tensor([lm.size], dtype=torch.float32) >= self.parameters['rel_dist'] and \
                    (tr_h - e_h) / torch.tensor([tr.size], dtype=torch.float32) >= self.parameters['rel_dist']:
                indirect_support = torch.max(indirect_support, torch.min(self.compute(tr, entity),
                                                                         self.compute(entity, lm)))
        return torch.max(direct_support, indirect_support)


class Near(Node):

    def __init__(self):
        self.params = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)

    def compute(self, tr, lm):

        if tr == lm:
            return 0
        connections = self.get_connections()
        raw_near_measure = Near_Raw.compute(tr, lm)
        raw_near_tr = [connections['near_raw'].compute(tr, entity) for entity in world.entities if
                       entity != tr]
        raw_near_lm = [connections['near_raw'].compute(lm, entity) for entity in world.entities if
                       entity != lm]
        avg_near = 0.5 * (np.average(raw_near_tr) + np.average(raw_near_lm))
        near_measure = raw_near_measure + (raw_near_measure - avg_near) * min(raw_near_measure, 1 - raw_near_measure)
        near_measure = torch.tensor([near_measure], dtype=torch.float32)
        if tr.compute_size() > lm.compute_size():
            near_measure -= self.params
        elif tr.compute_size() < lm.compute_size():
            near_measure += self.params
        return near_measure


class On(Node):
    def compute(self, tr, lm):
        if tr == lm:
            return torch.tensor([0], dtype=torch.float32)
        proj_dist = np.linalg.norm(
            np.array([tr.location[0] - lm.location[0],tr.location[1] - lm.location[1]]))
        proj_dist_scaled = proj_dist / (max(tr.size, lm.size) + 0.01)
        hor_offset = math.e ** (-0.3 * proj_dist_scaled)
        ret_val = Touching.compute(tr, lm) * Above.compute(tr, lm) \
            if hor_offset < 0.8 else Above.compute(tr, lm)

        if lm.get('planar') is not None and self.connections['larger_than'].compute(lm, tr) and \
                tr.centroid[2] > 0.5 * tr.dimensions[2]:
            ret_val = torch.max(ret_val, Touching.compute(tr, lm))

        for ob in lm.components:
            ob_ent = Entity(ob)
            if ob.get('working_surface') is not None or ob.get('planar') is not None:
                cmp1 = torch.tensor(
                    [0.5 * (v_offset(tr, ob_ent) + self.connections['projection_intersection'].compute(tr, ob_ent))],
                    dtype=torch.float32)
                cmp2 = torch.tensor(
                    [0.5 * (int(near(tr, ob_ent) > 0.99) + self.connections['larger_than'].compute(ob_ent, tr))],
                    dtype=torch.float32)
                ret_val = torch.max(ret_val, cmp1)
                ret_val = torch.max(ret_val, cmp2)
        if lm.get('planar') is not None and isVertical(lm):
            cmp = torch.tensor([math.exp(- 0.5 * get_planar_distance_scaled(tr, lm))],
                               dtype=torch.float32)
            ret_val = torch.max(ret_val, cmp)
        return ret_val

    def str(self):
        return 'on.p'


class Over(Node):
    def __init__(self):
        self.parameters = {"above_weight": torch.tensor(0.5, dtype=torch.float32, requires_grad=True),
                           "projection_intersection_weight": torch.tensor(0.2, dtype=torch.float32, requires_grad=True),
                           "near_weight": torch.tensor(0.3, dtype=torch.float32, requires_grad=True)}

    def compute(self, tr, lm):
        return self.parameters["above_weight"] * Above.compute(tr, lm) \
               + self.parameters["projection_intersection_weight"] * self.connections['projection_intersection'].compute(tr, lm) \
               + self.parameters["near_weight"]

    def str(self):
        return 'over.p'


class Under(Node):

    def compute(self, tr, lm):
        return self.connections['on'].compute(lm, tr)


class AtSameHeight(Node):
    def compute(self, tr, lm):
        dist = np.linalg.norm(tr.centroid[2] - lm.centroid[2])
        scaled_dist = dist / (tr.size + lm.size + 0.01)
        return math.e ** (-scaled_dist)


class At(Node):

    def compute(self, tr, lm):
        if tr == lm:
            return 0
        touching = self.connections['touching'].compute(tr, lm)
        at_same_height = self.connections['at_same_height'].compute(tr, lm)
        return at_same_height * touching if touching > 0.9 else at_same_height * self.connections['near'].compute(tr,
                                                                                                                  lm)


class Inside(Node):
    def compute(self, tr, lm):
        # a_bbox = a.bbox
        # b_bbox = b.bbox
        shared_volume = get_bbox_intersection(tr, lm)
        proportion = shared_volume / lm.volume
        return sigmoid(proportion, 1.0, 1.0)


class Between(Node):

    def __init__(self):
        self.parameters = torch.tensor([-0.05], dtype=torch.float32, requires_grad=True)

    def compute(self, tr, lm1, lm2):
        tr_to_lm1 = lm1.centroid - tr.centroid
        tr_to_lm2 = lm2.centroid - tr.centroid

        cos = np.dot(tr_to_lm1, tr_to_lm2) / (np.linalg.norm(tr_to_lm1) * np.linalg.norm(tr_to_lm2) + 0.001)

        scaled_dist = np.linalg.norm(lm1.centroid - lm2.centroid) / (2 * tr.size)

        dist_coeff = math.exp(self.parameters * torch.tensor([scaled_dist], dtype=torch.float32))

        ret_val = math.exp(- math.fabs(-1 - cos)) * dist_coeff
        return ret_val


if __name__ == '__main__':

    net = CustomNet()

    train_size = 1000
    batch_size = 20
    epochs = 10000

    samples = []
    labels = []
    for i in range(train_size):
        samples.append(Sample(i))
        labels.append(torch.tensor(i, dtype=torch.float32))

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    idx = 0
    for epoch in range(epochs):
        batch_loss = 0
        for j in range(batch_size):
            optimizer.zero_grad()
            outputs = net.compute(samples[idx])
            batch_loss += torch.sum((outputs - labels[idx]).pow(2))
            idx = idx + 1 if idx < train_size - 1 else 0

        batch_loss /= batch_size
        batch_loss.backward()
        optimizer.step()
        print("Epoch: %d, loss: %.3f" % (epoch, batch_loss))
        print("Params: ", net.params)
