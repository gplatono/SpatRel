from PIL import Image, ImageDraw
import sys

image_path = sys.argv[1]

with open("projections", "r") as f:
	for line in f.readlines():
		line = line.strip().split(":")
		if ("Wall" in line[0]) or ("Floor" in line[0]) or ("Ceiling" in line[0]):
			continue
		coor = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
		#print (line, coor)
		# coor[0] = 1920 * max(0, coor[0])
		# coor[1] = 1080 * (1 - max(0, coor[1]))
		# coor[2] = 1920 * min(1, coor[2])
		# coor[3] = 1080 * (1 - min(1, coor[3]))
		print (line[0], coor)
		img = Image.open(image_path) 
		img1 = ImageDraw.Draw(img)  
		img1.rectangle(coor, outline ="red")
		img.show()
		input()

		 


# scale_x = 100
# scale_y = 100
# tr_x = 9.6
# tr_y = 8.4
# shape = [scale_x*(1.528+tr_x), scale_y*(0.5+tr_y), scale_x*(3.113+tr_x), scale_y*(-1.15+tr_y)] 
# shape1 = [scale_x*(-7.5+tr_x), scale_y*(0.6+tr_y), scale_x*(-5.83+tr_x), scale_y*(-2.49+tr_y)]
# creating new Image object 

# create rectangle image 
#img1.rectangle(shape1, outline ="red") 
