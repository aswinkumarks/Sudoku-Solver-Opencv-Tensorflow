import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle

class CnnClassifier:
	def __init__(self,load_model=False):
		if load_model:
			self.load_model()

	def create_model(self):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Flatten())

		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

		model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
		model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
		self.model =model
	
	def load_data(self):
		mnist = tf.keras.datasets.mnist 
		(x_train, y_train),(x_test, y_test) = mnist.load_data()

		# labels_dirs = os.listdir('./Data')
		# images = []
		# image_labels = []
		# for label_name in labels_dirs:
		# 	files = os.listdir('./Data/'+label_name)
		# 	# vec = [0]*10
		# 	# vec[int(label_name)] = 1
		# 	for f_name in files:
		# 		img = cv2.imread('./Data/'+label_name+'/'+f_name,2)
		# 		img = cv2.resize(img,(64,64))
		# 		# print(img.shape)
		# 		# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# 		ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		# 		inv = cv2.bitwise_not(bw_img)
		# 		# print(inv.shape)
		# 		# cv2.imshow("image",inv)
		# 		# cv2.waitKey(0)
		# 		images.append(inv)
		# 		image_labels.append(int(label_name))

		# images = np.array(images)
		# image_labels = np.array(image_labels)
		# # np.random.shuffle(images)
		# # np.random.shuffle()
		# images,image_labels = shuffle(images,image_labels)
		# print(images.shape)
		# print(image_labels.shape)
		# print(x_train.shape,y_train.shape)
		# print(y_train[0])
		# self.x_train = tf.keras.utils.normalize(images[:9000], axis=1)
		self.x_train = tf.keras.utils.normalize(x_train, axis=1)
		# self.x_test = tf.keras.utils.normalize(images[9000:], axis=1)
		self.x_test = tf.keras.utils.normalize(x_test, axis=1)

		# self.y_train = image_labels[:9000]
		self.y_train = y_train
		# self.y_test = image_labels[9000:]
		self.y_test = y_test
		# exit(0)

	def save_model(self):
		self.model.save('./weights')

	def load_model(self):
		self.model = tf.keras.models.load_model('./weights')

	def evaluate(self):
		test_loss, test_acc = self.model.evaluate(x=self.x_test, y=self.y_test) 
		print("Test loss:",test_loss)
		print("Test accuracy:",test_acc)

	def train(self,no_epochs=10,save_model=True):
		self.create_model()
		self.load_data()
		self.model.fit(x=self.x_train, y=self.y_train, epochs=no_epochs)
		self.evaluate()
		if save_model:
			self.save_model()

	def detect(self,image):
		image = np.array([image])
		image = tf.keras.utils.normalize(image, axis=1)
		prediction = self.model.predict(image)
		return np.argmax(prediction)


if __name__ == '__main__':
	classifier = CnnClassifier()
	classifier.load_data()
	# res = classifier.detect(classifier.x_train[0])
	# print(res)
	classifier.train()
	# classifier.evaluate()

	