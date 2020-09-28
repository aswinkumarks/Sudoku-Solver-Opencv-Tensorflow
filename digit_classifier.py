import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import sys

class CnnClassifier:
	def __init__(self,model_name=None):
		if model_name is not None:
			self.load_model(model_name)

	def create_model(self,learning_rate=1e-3):
		print("[INFO] Creating the model")
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(32, (5, 5), padding="same",
			input_shape=(28,28,1)))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

		# CONV => RELU => POOL layers
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

		# FC => RELU layers
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(64))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.Dropout(0.5))

		# FC => RELU layers
		model.add(tf.keras.layers.Dense(64))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.Dropout(0.5))

		# softmax classifier
		model.add(tf.keras.layers.Dense(10))
		model.add(tf.keras.layers.Activation("softmax"))

		model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
						,metrics=["accuracy"])
		self.model =model
	
	def load_data(self):
		print("[INFO] Loading Data")
		mnist = tf.keras.datasets.mnist 
		(x_train, y_train),(x_test, y_test) = mnist.load_data()

		x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
		x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

		# scale data to the range of [0, 1]
		x_train = x_train.astype("float32") / 255.0
		x_test = x_test.astype("float32") / 255.0

		# convert the labels from integers to vectors
		le = LabelBinarizer()
		y_train = le.fit_transform(y_train)
		y_test = le.transform(y_test)

		self.x_train = x_train
		self.x_test = x_test

		self.y_train = y_train
		self.y_test = y_test

	def save_model(self,model_name):
		print("[INFO] Saving the model")
		self.model.save(model_name, save_format="h5")

	def load_model(self,model_name='trained_model.h5'):
		print("[INFO] Loading model")
		self.model = tf.keras.models.load_model(model_name)

	def evaluate(self):
		test_loss, test_acc = self.model.evaluate(x=self.x_test, y=self.y_test) 
		print("Test loss:",test_loss)
		print("Test accuracy:",test_acc)

	def train(self,model_name,batch_size=128,no_epochs=10,learning_rate=1e-3,save_model=True):
		self.create_model(learning_rate=learning_rate)
		self.load_data()
		print("[INFO] Training model")
		self.model.fit(self.x_train, self.y_train,validation_data=(self.x_test, self.y_test),
					batch_size=batch_size,epochs=no_epochs,verbose=1)
		self.evaluate()
		if save_model:
			self.save_model(model_name)

	def detect(self,image):
		assert image.shape[0]==28 and image.shape[1]==28
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		prediction = self.model.predict(image).argmax(axis=1)[0]
		return prediction


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='--train model_name')
	parser.add_argument('--batch_size',default=128,type=int,
						help='--batch_size 128')
	parser.add_argument('--epochs',default=5, type=int,
						help='--epochs 10')
	parser.add_argument('--lr',default=0.01, type=float,
						help='--lr 0.1')
	parser.add_argument('--evaluate', help='--evaluate model_name.h5')

	if not len(sys.argv) > 1:
		parser.print_help()
		sys.exit(0)
	else:
		args = parser.parse_args()

	if args.evaluate is not None:
		classifier = CnnClassifier(model_name=args.evaluate)
		classifier.load_data()
		classifier.evaluate()

	if args.train is not None:
		classifier = CnnClassifier()
		classifier.train(model_name=args.train,batch_size=args.batch_size,
							 no_epochs=args.epochs,learning_rate= args.lr)


	