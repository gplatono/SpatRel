data_dict = {}
with open("annotation_dump", 'r') as file:
	data = [('scene_name=' + line).split(";") for line in file.readlines()]
	print (data)	
	for annotation in data:
		local_dict = {}
		for item in annotation:
			item = item.split('=')
			local_dict[item[0]] = item[1] if ":" not in item[1] else item[1].split(":")
		print(local_dict)
		relatum = local_dict['relatum']
		scene_name = local_dict['scene_name']
		file_name = local_dict['scene_name'].split(".") + "_test.data"

		input()

