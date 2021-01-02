import numpy as np
from numpy import linalg as LA
from skimage import io
from skimage import color as cl
from skimage.restoration import denoise_bilateral
from math import sqrt, exp
import cv2
import matplotlib.pyplot as plt

class SuperPixel:

	def __init__(self, sId, assoc=0):
		self.sId = sId
		self.assoc = assoc 

class AbstractionResampler:

	def __init__(self, img, output_width, output_height):
		self.nb_colors = 8

		self.input_height, self.input_width, _ = img.shape

		self.output_width = output_width
		self.output_height = output_height

		self.input_area = (self.input_height * self.input_width)
		self.output_area = (self.output_height * self.output_width)

		self.img = img
		self.lab_img = cl.rgb2lab(img)

		self.output_lab = np.zeros([self.output_height, self.output_width, 3])

		self.converged = False
		self.palette_maxed = False
		self.iteration = 0
		self.range_ = sqrt(self.input_area / self.output_area)

		# Vector of superpixels
		self.superpixels = []
		#self.superpixels.clear()

		# Matrix of int
		self.pixel_map = np.zeros([self.input_height, self.input_width])
		#self.pixel_map.clear()
		
		sx = self.input_width / self.output_width
		sy = self.input_height / self.output_height

		for i in range(self.output_width):
			for j in range(self.output_height):
				p = SuperPixel(i * self.output_width + j)
				p.position = np.array([(j + 1/2) * sy , (i + 1/2) * sx])
				p.color = [0, 0, 0]
				self.superpixels.append(p)

				for x in range(round(i * sx), round((i + 1) * sx)):
					for y in range(round(j * sy), round((j + 1) * sy)):
						self.pixel_map[y][x] = p.sId

		self.update_superpixels() #TODO

		# Init palette
		color = np.array([0.0, 0.0, 0.0])
		for sp in self.superpixels:
			color += sp.color

		color /= self.output_area
		self.palette = [color]

		# Init MCDA
		self.prob_o = 1.0 / self.output_area
		self.prob_c = [0.5, 0.5]
		self.prob_co = np.full((self.output_area, 2), 0.5)

		self.palette.append(color + 0.8 * self.get_max_eigen(0)[0])
		self.sub_superpixels = [[0, 1]]

		self.temperature = 1.1 * sqrt(2 * self.get_max_eigen(0)[1])

	def resample(self):
		while not(self.is_done()):
			self.iterate()

		self.finalize()

	def is_done(self):
		if(self.iteration > 100):
			return True

		return self.converged

	def iterate(self):
		self.remap_pixels()
		self.update_superpixels()

		self.associate_superpixels()
		err = self.refine_palette()

		if(err < 1.0):
			if (self.temperature <= 1.0):
				self.converged = True
			else:
				self.temperature = max(1.0, 0.7 * self.temperature)

			self.expand_palette()

	def update_superpixels(self):
		count = []
		# Do a copy
		tmp_superpixels = np.array(self.superpixels)

		for sp in self.superpixels:
			sp.position = np.array([0.0, 0.0])
			sp.color = [0, 0, 0]
			count.append(0)

		for i in range(self.input_width):
			for j in range(self.input_height):
				idx = int(self.pixel_map[j][i])
				self.superpixels[idx].position += [j, i]# / self.input_width, j / self.input_height]
				self.superpixels[idx].color += self.lab_img[j, i]
				count[idx] += 1

		for i in range(len(self.superpixels)):
			if count[i]:
				self.superpixels[i].position /= count[i]
				self.superpixels[i].color /= count[i]
			else:
				print("???")
				self.superpixels[i].position = tmp_superpixels[i].position
				x = int(self.superpixels[i].position[1] * self.input_width)
				y = int(self.superpixels[i].position[0] * self.input_height)
				self.superpixels[i].color = self.lab_img[y, x]
		
		# Position smoothing
		'''p = newp = []

		for i in range(self.output_width):
			col = newcol = []
			for j in range(self.output_height):
				col.append(self.superpixels[i * self.output_width + j].position)
				newcol.append([0, 0])
			p.append(col)
			newp.append(newcol)
		'''
		newp = np.zeros((self.output_height, self.output_width, 2))

		for i in range(self.output_width):
			for j in range(self.output_height):
				c = 0
				if i > 0:
					c += 1
					newp[j][i] += self.superpixels[(i - 1) * self.output_width + j].position
				if j > 0:
					c += 1
					newp[j][i] += self.superpixels[i * self.output_width + j - 1].position
				if i < self.output_width - 1:
					c += 1
					newp[j][i] += self.superpixels[(i + 1) * self.output_width + j].position
				if j < self.output_height - 1:
					c += 1
					newp[j][i] += self.superpixels[i * self.output_width + j + 1].position
				newp[j][i] /= c

		for i in range(self.output_width):
			for j in range(self.output_height):
				self.superpixels[i * self.output_width + j].position = 0.6 *\
																		self.superpixels[j * self.output_height + i].position +\
																		0.4 * newp[j][i]

		# Color smoothing
		c = np.zeros([self.output_height, self.output_width, 3])
		for i in range(self.output_width):
			for j in range(self.output_height):
				c[j][i] = self.superpixels[i * self.output_width + j].color

		# Impossible d'effectuer un bilateral filter sur une image en L*a*b
		# -> Le faire sur une image en grayscale
		c = cl.lab2rgb(c)
		c = denoise_bilateral(c, multichannel=True)
		c = cl.rgb2lab(c)

		for i in range(self.output_width):
			for j in range(self.output_height):
				self.superpixels[i * self.output_width + j].color = c[j][i]

	def remap_pixels(self):
		dmap = np.full((self.input_height, self.input_width), -1)
		averaged_palette = self.get_averaged_palette()

		for (k, sp) in enumerate(self.superpixels):
			x = sp.position[1] * self.input_width
			y = sp.position[0] * self.input_height
			x0 = max(0, int(x - self.range_))
			y0 = max(0, int(y - self.range_))
			x1 = min(self.input_width, int(x + self.range_))
			y1 = min(self.input_height, int(y + self.range_))

			color = averaged_palette[sp.assoc]

			for i in range(x0, x1):
				for j in range(y0, y1):
					d = self.slic_distance(i, j, [x, y], color)
					if dmap[j][i] > d or dmap[j][i] < 0:
						dmap[j][i] = d
						self.pixel_map[j][i] = k

	def get_averaged_palette(self):
		averaged_palette = np.copy(self.palette)

		if not self.palette_maxed:
			for sb in self.sub_superpixels:
				idx_1 = sb[0]
				idx_2 = sb[1]
				color_1 = self.palette[idx_1]
				color_2 = self.palette[idx_2]
				weight_1 = self.prob_c[idx_1]
				weight_2 = self.prob_c[idx_2]
				total_weight = weight_1 + weight_2

				average_color = [weight_1 * color_1[0] + weight_2 * color_2[0],\
									weight_1 * color_1[1] + weight_2 * color_2[1],\
									weight_1 * color_1[2] + weight_2 * color_2[2]]
				averaged_palette[idx_1] = average_color
				averaged_palette[idx_2] = average_color

		return averaged_palette

	def slic_distance(self, i, j, pos, spcolor):
		dx = i - pos[0]
		dy = j - pos[1]
		color = self.lab_img[j, i]
		color_error = LA.norm(color - spcolor)
		dist_error = LA.norm(dx - dy)

		return color_error + 45.0 / self.range_ * dist_error

	def associate_superpixels(self):
		palette_size = len(self.palette)
		new_prob_c = np.zeros(palette_size)
		self.prob_co = np.zeros([self.output_width * self.output_height, palette_size])

		overT = -1.0 / self.temperature

		for k in range(len(self.superpixels)):
			best_index = palette_size
			probs = []
			pixel = self.superpixels[k].color
			sum_prob = 0

			for i in range(palette_size):
				# Norm between colors
				color_error = LA.norm(self.palette[i] - pixel)
				prob = self.prob_c[i] * exp(color_error * overT)
				sum_prob += prob
				probs.append(prob)

				if (best_index == palette_size or color_error < best_error):
					best_index = i
					best_error = color_error

			self.superpixels[k].assoc = best_index

			for i in range(palette_size):
				norm_prob = probs[i] / sum_prob
				self.prob_co[k][i] = norm_prob
				new_prob_c[i] += norm_prob * self.prob_o

		self.prob_c = new_prob_c
		
	def refine_palette(self):
		color_sums = np.zeros((len(self.palette), 3))

		for k in range(len(self.superpixels)):
			for i in range(len(self.palette)):
				color_sums[i] += self.superpixels[k].color * self.prob_co[k][i] * self.prob_o

		palette_error = 0
		for i in range(len(color_sums)):
			color = self.palette[i]
			new_color = color_sums[i] / self.prob_c[i]
			palette_error += LA.norm((color - new_color))
			self.palette[i] = new_color

		return palette_error

	def expand_palette(self):
		if self.palette_maxed:
			return

		# Splits = cluster_distance, index
		splits = []
		nb_sub_superpixels = len(self.sub_superpixels)
		for (i, sb) in enumerate(self.sub_superpixels):
			idx_1 = sb[0]
			idx_2 = sb[1]
			color_1 = self.palette[idx_1]
			color_2 = self.palette[idx_2]

			cluster_error = LA.norm((color_1 - color_2))
			if (cluster_error > 1.6):
				splits.append([cluster_error, i])
			else:
				self.palette[idx_2] += self.get_max_eigen(idx_1)[0] * 0.8
				
		splits.sort()
		for s in splits:
			self.split_color(s[1])
			if len(self.palette) >= 2 * self.nb_colors:
				self.condense_palette()
				break

	def split_color(self, idx):
		idx_1 = self.sub_superpixels[idx][0]
		idx_2 = self.sub_superpixels[idx][1]

		next_index1 = len(self.palette)
		next_index2 = len(self.palette) + 1

		color_1 = self.palette[idx_1]
		color_2 = self.palette[idx_2]

		sub_color_1 = color_1 + self.get_max_eigen(idx_1)[0] * 0.8
		sub_color_2 = color_2 + self.get_max_eigen(idx_2)[0] * 0.8

		self.palette.append(sub_color_1)
		self.sub_superpixels[idx][1] = next_index1
		self.prob_c[idx_1] *= 0.5
		#self.prob_c.append(self.prob_c[idx_1])
		#self.prob_co.append(self.prob_co[idx_1])		

		self.prob_c = np.append(self.prob_c, self.prob_c[idx_1])
		self.prob_co = np.c_[self.prob_co, self.prob_co[:, idx_1]]

		self.palette.append(sub_color_2)
		new_pair = [idx_2, next_index2]
		self.sub_superpixels.append(new_pair)
		self.prob_c[idx_2] *= 0.5
		#self.prob_c.append(self.prob_c[idx_2])
		#self.prob_co.append(self.prob_co[idx_2])
		self.prob_c = np.append(self.prob_c, self.prob_c[idx_2])
		self.prob_co = np.c_[self.prob_co, self.prob_co[:, idx_2]]

	def condense_palette(self):
		self.palette_maxed = True

		new_palette = []
		new_prob_c = np.zeros(self.prob_c.shape)
		new_prob_co = np.zeros(self.prob_co.shape)

		for (j, sb) in enumerate(self.sub_superpixels):
			idx_1 = sb[0]
			idx_2 = sb[1]

			weight_1 = self.prob_c[idx_1]
			weight_2 = self.prob_c[idx_2]

			total_weight = weight_1 + weight_2

			new_palette.append(self.palette[idx_1] * weight_1 + self.palette[idx_2] * weight_2)
			new_prob_c[j] = self.prob_c[idx_1] + self.prob_c[idx_2]
			new_prob_co[:, j] = self.prob_co[:, idx_1]

			for sp in self.superpixels:
				if sp.assoc == idx_1 or sp.assoc == idx_2:
					sp.assoc = j

		self.palette = new_palette
		self.prob_c = new_prob_c
		self.prob_co = new_prob_c

	def finalize(self):
		averaged_palette = self.get_averaged_palette()
		for i in range(self.output_width):
			for j in range(self.output_height):
				sp = self.superpixels[i * self.output_width + j]
				self.output_lab[j, i] = averaged_palette[sp.assoc]
				self.output_lab[j, i][1] *= 1.1
				self.output_lab[j, i][2] *= 1.1

		self.output_lab = cl.lab2rgb(self.output_lab)

	def visualizeSuperpixel(self):
		n_neighbors = 8
		dx = [-1, -1, 0, 1, 1, 1, 0, -1]
		dy = [0, -1, -1, -1, 0, 1, 1, 1]

		output = self.img
		for idx in range(len(self.palette)):
			for i in range(self.input_width):
				for j in range(self.input_height):
					output[j][i] = cl.lab2rgb(self.palette[idx])

		for i in range(self.output_width):
			for j in range(self.output_height):
				idx = self.pixel_map[j][i]
				cnt = 0

				for k in range(n_neighbors):
					x = i + dx[k]
					y = j + dy[k]
					if x >= 0 and x < self.output_width and\
						y >= 0 and y > self.output_height and\
						self.pixel_map[y][x] != id:
						cnt += 1

				if cnt > 1:
					output[j][i] = [0, 0, 255]

		for i in range(len(self.superpixels)):
			sp = self.superpixels[i]
			x = int(sp.position[0] * self.output_width)
			y = int(sp.position[1] * self.output_height)
			output[y][x] = [0, 255, 0]

			for k in range(n_neighbors):
				xx = x + dx[k]
				yy = y + dy[k]

				if xx >= 0 and xx < self.output_width and\
					yy >= 0 and yy > self.output_height:
					output[yy][xx] = [0, 255, 0]

		return output
		
	def get_max_eigen(self, pidx):
		sum_ = 0
		matrix = np.zeros((3,3))

		for y in range(self.output_height):
			for x in range(self.output_width):
				
				prob_oc = self.prob_co[y * self.output_height + x][pidx]\
								* self.prob_o / self.prob_c[pidx]

				sum_ += prob_oc
				color_error = abs(self.palette[pidx] - self.superpixels[y * self.output_height + x].color)

				matrix[0, 0] += prob_oc * color_error[0] * color_error[0]
				matrix[0, 1] += prob_oc * color_error[1] * color_error[0]
				matrix[0, 2] += prob_oc * color_error[2] * color_error[0]
				matrix[1, 0] += prob_oc * color_error[0] * color_error[1]
				matrix[1, 1] += prob_oc * color_error[1] * color_error[1]
				matrix[1, 2] += prob_oc * color_error[2] * color_error[1]
				matrix[2, 0] += prob_oc * color_error[0] * color_error[2]
				matrix[2, 1] += prob_oc * color_error[1] * color_error[2]
				matrix[2, 2] += prob_oc * color_error[2] * color_error[2]
		
		values, vectors = LA.eig(matrix)
		eVec = np.array([vectors[0][0], vectors[0][1], vectors[0][2]])

		len_ = LA.norm(eVec)
		if len_ > 0:
			eVec *= (1.0 / len_)
		eVal = abs(values[0])

		return [eVec, eVal]

# img = io.imread("mario.jpg")
# #img = io.imread("bridge_1.jpg")

# sampler = AbstractionResampler(img, 225, 225)
# sampler.resample()

# plt.figure()
# io.imshow(sampler.output_lab)
# plt.show()