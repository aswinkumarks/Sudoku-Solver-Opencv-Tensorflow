import numpy as np
import cv2
from skimage.segmentation import clear_border
from scipy.spatial import distance
from digit_classifier import CnnClassifier

class Sudoku_Detector:
	def __init__(self,debug=False):
		self._debug = debug
		self.classifier = CnnClassifier(load_model=True)

	def preprocess(self,image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 3)

		thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		thresh = cv2.bitwise_not(thresh)

		if self._debug:
			cv2.imshow("After Preprocessing",thresh)
			cv2.waitKey(0)

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
		# self.puzzle = four_point_transform(self.original_image, outer_square.reshape(4, 2))
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
		thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		thresh = clear_border(thresh)
		cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		if len(cnts) == 0:
			return 0

		cnt = max(cnts, key=cv2.contourArea)
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [cnt], -1, 255, -1)

		(h, w) = thresh.shape
		percentFilled = cv2.countNonZero(mask) / float(w * h)

		if percentFilled < 0.03: # Noise
			return 0

		digit = cv2.bitwise_and(thresh, thresh, mask=mask)
		digit = cv2.resize(digit,(28,28))
		# thresh = cv2.resize(thresh,(28,28))

		res = self.classifier.detect(digit)
		if self._debug:
			print("Detected Digit: ",res)
			cv2.imshow("cell",digit)
			cv2.waitKey(0)

		return res

	def extract_puzzle(self,image):
		step_size = int(image.shape[0]/9)
		width, height = image.shape[:2]
		puzzle = []
		for x in range(0,width,step_size):
			row = []
			for y in range(0,height,step_size):
				cell_top_left = (x,y)
				cell_bottom_rigth = (x+step_size,y+step_size)
				digit = self.extract_digits(image[x:x+step_size,y:y+step_size])
				row.append(digit)

			puzzle.append(row)

		return puzzle

	def detect(self,image):
		self.original_image = image.copy()
		proc_img, gray = self.preprocess(image)
		corners = self.find_corners(proc_img)

		gray = self.wrap_perspective(gray,corners)
		self.cropped_image  = self.wrap_perspective(image,corners)
		# proc_img = self.wrap_perspective(proc_img,corners)

		puzzle = self.extract_puzzle(gray)
		return puzzle

	def visualize_solution(self,puzzle):
		image = self.cropped_image
		step_size = int(image.shape[0]/9)
		width, height = image.shape[:2]
		for i,y in zip(range(9),range(0,width,step_size)):
			for j,x in zip(range(9),range(0,height,step_size)):
				cell_top_x,cell_top_y = (x,y)
				cell_bottom_x,cell_bottom_y = (x+step_size,y+step_size)
				textX = int((cell_bottom_x - cell_top_x) * 0.30)
				textY = int((cell_bottom_y - cell_top_y) * -0.3)
				textX += cell_top_x
				textY += cell_bottom_y
				# print(textX,textY)
				cv2.putText(image, str(puzzle[i][j]), (textX, textY),
					cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2,cv2.LINE_AA)

		cv2.imshow("Solution",image)
		cv2.waitKey(0)

if __name__ == "__main__":
	img = cv2.imread("./sudoku.jpg")
	# img = cv2.imread("./sudoku-original.jpg")
	detector = Sudoku_Detector()
	puzzle = detector.detect(img)
	print(puzzle)