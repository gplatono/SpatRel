import os
import sys
import glob
import subprocess
import json
import copy

path_to_data = sys.argv[1]

scene_path = path_to_data + os.sep + "scenes"
ann_path = path_to_data + os.sep + "annotations"
 
scenes = sorted(glob.glob(scene_path + os.sep + "*.blend"))
annotations = sorted(glob.glob(ann_path + os.sep + "*.data"))

train_subset = ["RW1.data", "RW2.data", "RW3.data", "RW4.data", "RW5.data", "RW6.data", "RW7.data", "RW8.data", "RW9.data", "RW10.data", "RW11.data", "RW12.data", "RW13.data", "RW14.data", "RW15.data"]
test_subset = ["RW101.data", "RW201.data", "RW202.data", "RW203.data", "RW204.data", "RW205.data", "RW206.data", "RW207.data", "RW208.data", "RW209.data", "RW210.data"]

def model_train(epochs):
	train_acc_dict = None
	scene_idx = 0
	actual_epoch = 0
	for epoch in range(epochs):
		scene = scenes[scene_idx]
		name = scene.split(os.sep)[-1].split(".blend")[0] + '.data'
		if ann_path + os.sep + name in annotations and name in train_subset:

			#command = ['/Applications/Blender.app/Contents/MacOS/Blender', scene, '-P', 'train_scene.py', '--', ann_path + os.sep + name]
			command = ['../blender/blender', scene, '--background', '-P', 'train_scene.py', '--', ann_path + os.sep + name, 'train']
			#command = ['blender', scene, '--background', '-P', 'train_scene.py', '--', ann_path + os.sep + name, 'train']
			subprocess.run(command)
			
			tmp_acc = None
			with open('rel_accuracies_train', 'r') as file:
				tmp_acc = json.load(file)

			train_acc_dict = update_dict(train_acc_dict, tmp_acc)
			# if tmp_acc is not None and tmp_acc != "":
			# 	if train_acc is None:
			# 		train_acc = copy.deepcopy(tmp_acc)
			# 		for key in rel_acc:
			# 			train_acc[key]['total'] = [train_acc[key]['total']]
			# 			train_acc[key]['acc'] = [train_acc[key]['acc']]
			# 	else:
			# 		for key in tmp_acc:
			# 			train_acc[key]['total'].append(tmp_acc[key]['total'])
			# 			train_acc[key]['acc'].append(tmp_acc[key]['acc'])

			# 	print ()
			# 	for key in tmp_acc:
			# 		print (key.upper() + ", {} annotations, accuracy: {:.3f}".format(tmp_acc[key]['total'], tmp_acc[key]['acc']))
			# 	print ()
				
			actual_epoch += 1

		scene_idx += 1
		if scene_idx == len(scenes):
			scene_idx = 0

	return train_acc_dict	

def model_evaluate(epochs):
	test_acc_dict = None
	scene_idx = 0
	actual_epoch = 0
	for epoch in range(epochs):
		scene = scenes[scene_idx]
		name = scene.split(os.sep)[-1].split(".blend")[0] + '.data'
		if ann_path + os.sep + name in annotations and name in test_subset:

			#command = ['/Applications/Blender.app/Contents/MacOS/Blender', scene, '-P', 'train_scene.py', '--', ann_path + os.sep + name]
			command = ['../blender/blender', scene, '--background', '-P', 'train_scene.py', '--', ann_path + os.sep + name, 'test']
			#command = ['blender', scene, '--background', '-P', 'train_scene.py', '--', ann_path + os.sep + name, 'test']
			subprocess.run(command)
			
			tmp_acc = None
			with open('rel_accuracies_test', 'r') as file:
				tmp_acc = json.load(file)

			test_acc_dict = update_dict(test_acc_dict, tmp_acc)				
			actual_epoch += 1

		scene_idx += 1
		if scene_idx == len(scenes):
			scene_idx = 0

	return test_acc_dict

def update_dict(main_dict, tmp_acc):
	if tmp_acc is not None and tmp_acc != "":
		if main_dict is None:
			main_dict = copy.deepcopy(tmp_acc)
			for key in main_dict:
				main_dict[key]['total'] = [main_dict[key]['total']]
				main_dict[key]['acc'] = [main_dict[key]['acc']]				
		else:
			for key in tmp_acc:
				main_dict[key]['total'].append(tmp_acc[key]['total'])
				main_dict[key]['acc'].append(tmp_acc[key]['acc'])
				main_dict[key]['tp'] += tmp_acc[key]['tp']
				main_dict[key]['tn'] += tmp_acc[key]['tn']
				main_dict[key]['fp'] += tmp_acc[key]['fp']
				main_dict[key]['fn'] += tmp_acc[key]['fn']

		print ()
		for key in tmp_acc:
			print (key.upper() + ", {} annotations, accuracy: {:.3f}".format(tmp_acc[key]['total'], tmp_acc[key]['acc']))
		print ()

	return main_dict

def print_stats(acc_dict):
	for key in acc_dict:
		total_non_zero = sum(list(map(lambda x: 1 if x != 0 else 0, acc_dict[key]['total'])))
		acc_dict[key]['acc'] = sum(acc_dict[key]['acc']) / total_non_zero if total_non_zero != 0 else 0
		tp = acc_dict[key]['tp']
		tn = acc_dict[key]['tn']
		fp = acc_dict[key]['fp']
		fn = acc_dict[key]['fn']
		if tp + fp > 0 and tp + fn > 0:		
			precision = tp / (tp + fp)		
			recall = tp / (tp + fn)
			total_accuracy = (tp + tn) / (tp + tn + fp + fn)
			F1 = 2 / (1 / precision + 1 / recall)
			print (key.upper() + ", {} total annontations, avg accuracy: {:.3f}, total accuracy: {:.3f}, tp: {}, tn: {}, fp: {}, fn: {}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(sum(acc_dict[key]['total']), acc_dict[key]['acc'], \
			total_accuracy, tp, tn, fp, fn, precision, recall, F1))
		else:
			print (key.upper() + ", {} total annontations, avg accuracy: {:.3f}, tp: {}, tn: {}, fp: {}, fn: {}".format(sum(acc_dict[key]['total']), acc_dict[key]['acc'], tp, tn, fp, fn))


if __name__ == "__main__":
	train_acc_dict = model_train(len(scenes))
	test_acc_dict = model_evaluate(len(scenes))

	print ("\nTRAINING SUMMARY:")	
	print_stats(train_acc_dict)

	print ("\nTESTING SUMMARY:")
	print_stats(test_acc_dict)