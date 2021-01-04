from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def reduce_color(img, n_colors):
	arr = img.reshape((-1, 3))
	kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
	labels = kmeans.labels_
	centers = kmeans.cluster_centers_
	less_colors = centers[labels].reshape(img.shape).astype('uint8')

	return less_colors

def nearest_neighbours_colors(img, palette):
	# colors should always be equal to 3
	w_in, h_in, colors = img.shape
	img_out = np.zeros((w_in, h_in, colors))
	palette_size = len(palette)

	tmp_color = {}
	for x in range(w_in):
		for y in range(h_in):
			if tuple(img[x][y]) not in tmp_color:
				tmp_color[tuple(img[x][y])] = 1		

	color_image = np.zeros((palette_size, colors))
	i = 0
	for key in tmp_color:
		color_image[i] = key
		i += 1

	color_distance = np.zeros((palette_size, palette_size))
	color_to_use = []
	palette_to_use = []

	for i in range(palette_size):
		for j in range(palette_size):
			color_distance[i, j] = np.linalg.norm(color_image[i] - palette[j])
		
		color_to_use.append(i)
		palette_to_use.append(i)
        
	print(color_distance)
    
	new_order = []
	
	while(len(new_order) != palette_size):
		minimum = color_distance[color_to_use[0], palette_to_use[0]]
		index = [color_to_use[0], palette_to_use[0]]
		for i in color_to_use:
			for j in palette_to_use:
				if color_distance[i, j] < minimum:
					minimum = color_distance[i, j]
					index = [i, j]

		new_order.append(index)
		# Remove i value from array
		color_to_use.remove(index[0])
		# Remove j value from array
		palette_to_use.remove(index[1])

	print(new_order)
    
	for x in range(w_in):
		for y in range(h_in):
			for c in new_order:
				if (img[x, y] == color_image[c[0]]).all():
					img_out[x, y] = palette[c[1]]
					#print(palette[c])

	return img_out / 255.0
		
def apply_texture(img, palette_hex):
	w_in, h_in, _ = img.shape

	# Convert list from hex to dec
	palette = np.zeros((len(palette_hex), 3))
	for hx in range(len(palette_hex)):
		palette[hx] = [int(palette_hex[hx][i + 1:i + 3], 16) for i in (0, 2, 4)] 

	dict_colors = {}
	for x in range(w_in):
		for y in range(h_in):
			if tuple(img[x][y]) not in dict_colors:
				dict_colors[tuple(img[x][y])] = 1

	if len(dict_colors) > len(palette):
		img = reduce_color(img, len(palette))
    
	img_out = nearest_neighbours_colors(img, palette)
	return img_out
	# plt.imshow(img_out)
	# plt.show()
		

# img = io.imread("mario_300.jpg")
# palette = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
# 			'#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ffffff']
# apply_texture(img, palette)