import numpy as np
import cv2
from skimage.segmentation import clear_border
import imutils
from scipy.spatial import distance
from network import MultilayerNeuralNetwork
from digit_classifier import CnnClassifier
from imutils.perspective import four_point_transform


class Sudoku_Detector:
	def __init__(self,debug=False):
		self._debug = debug
		# self.classifier = MultilayerNeuralNetwork(model="./trained_weights.npz")
		self.classifier = CnnClassifier(load_model=True)


	def preprocess(self,image):
		# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		# proc = cv2.GaussianBlur(gray, (9, 9), 0)
		# proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

		# proc = cv2.bitwise_not(proc, proc)  
		# kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
		# proc = cv2.dilate(proc, kernel)

		# proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# proc = cv2.GaussianBlur(proc, (7, 7), 0)  
		# proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) 
		# proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 19, 2)
		# proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, np.ones((2, 2)))

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 3)

		thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		thresh = cv2.bitwise_not(thresh)

		if self._debug:
			cv2.imshow("After Preprocessing",thresh)
			cv2.waitKey(0)
			# exit(0)

		return thresh,gray


	def find_corners(self,image):
		# cnts,_ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts,_ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		outer_square = None
		for cnt in cnts:
			peri = cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt,0.01*peri,True)
			if len(approx) == 4:
				outer_square = approx
				break

		if outer_square is None:
			print("Could not find the grid")
			exit(0)
		# outer_square = max(cnts,key=cv2.contourArea)
		# peri = cv2.arcLength(outer_square, True)
		# outer_square = cv2.approxPolyDP(outer_square,0.01*peri,True)
		self.puzzle = four_point_transform(self.original_image, outer_square.reshape(4, 2))
		# warped = four_point_transform(gray, outer_square.reshape(4, 2))

		if self._debug:
			clone = self.original_image.copy()
			cv2.drawContours(clone,outer_square, -1, (0, 255, 0), 5)
			cv2.imshow('Corners',clone)
			cv2.waitKey(0)

		outer_points = [list(point[0]) for point in outer_square] 
		top_left,top_right = min(outer_points,key=lambda x:x[0]+x[1]), max(outer_points,key=lambda x:x[0]-x[1])
		bottom_left,bottom_right = max(outer_points,key=lambda x:x[1]-x[0]), max(outer_points,key=lambda x:x[1]+x[0])
		points = np.array([top_left,top_right,bottom_right,bottom_left],dtype='float32')

		return points


	def wrap_perspective(self,image,points):
		dists = [distance.euclidean(points[i],points[i-1]) for i in range(1,4)]
		dists.append(distance.euclidean(points[0],points[3]))
		side = 9 * round(max(dists)/9)
		new_points = np.array([[0,0], [side,0], [side,side], [0,side]],dtype='float32')

		tran_mat = cv2.getPerspectiveTransform(points, new_points)
		image = cv2.warpPerspective(image, tran_mat, (int(side), int(side)))

		if self._debug:
			cv2.imshow("ROI",image)
			cv2.waitKey(0)
			

		return image


	def extract_digits(self,cell):
		# cell = cv2.GaussianBlur(cell, (3, 3), 0)
		# cell = cv2.resize(cell,(28,28),interpolation=cv2.INTER_AREA)
		thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		thresh = clear_border(thresh)

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		if len(cnts) == 0:
			return None

		cnt = max(cnts, key=cv2.contourArea)
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [cnt], -1, 255, -1)

		(h, w) = thresh.shape
		percentFilled = cv2.countNonZero(mask) / float(w * h)

		if percentFilled < 0.03: # Noise
			return None

		digit = cv2.bitwise_and(thresh, thresh, mask=mask)
		digit = cv2.resize(digit,(64,64),interpolation=cv2.INTER_AREA)
		thresh = cv2.resize(thresh,(28,28),interpolation=cv2.INTER_AREA)
		# thresh.re
		# print(type(digit))
		# print(digit.shape)
		res = self.classifier.detect(digit)
		# res = self.classifier.predict(thresh.reshape(28*28,-1))
		# res = self.classifier.predict(digit)
		# res = np.argmax(res,axis=0)
		print(res)
		# print(res.shape)
		return digit
		# return thresh

	def get_cells(self,image):
		# image = cv2.GaussianBlur(image, (9, 9), 0)
		step_size = int(image.shape[0]/9)
		width, height = image.shape[:2]
		cells = []
		for x in range(0,width,step_size):
			for y in range(0,height,step_size):
				cell_top_left = (x,y)
				cell_bottom_rigth = (x+step_size,y+step_size)
				cells.append((cell_top_left,cell_bottom_rigth))
				cell = self.extract_digits(image[x:x+step_size,y:y+step_size])
				if self._debug:
					if cell is not None:
						cv2.imshow("cell",cell)
						cv2.waitKey(0)

		# print(cells)
		return cells

	def detect(self,image):
		self.original_image = image.copy()
		proc_img, gray = self.preprocess(image)
		corners = self.find_corners(proc_img)

		gray = self.wrap_perspective(gray,corners)
		proc_img = self.wrap_perspective(proc_img,corners)

		self.get_cells(gray)

		print(image.shape)


if __name__ == "__main__":
	img = cv2.imread("./sudoku.jpg")
	# img = cv2.imread("./sudoku-original.jpg")
	detector = Sudoku_Detector(debug=True)
	detector.detect(img)