from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
from threading import Thread
from scipy.spatial import distance as dist


NUM_DATA=2
measures = np.zeros([1,NUM_DATA])
first_time=True



class WebcamVideoStream:
	def __init__(self, src):

		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		self.stopped = False
	def start(self):

		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		while True:
			if self.stopped:
				return

			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		return self.frame
	def stop(self):
		self.stopped = True


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
# vs = WebcamVideoStream(src=0).start()

cap = cv2.VideoCapture(0)

fps = FPS().start()
init_time=time.time()
index=0

while True:

	# Descomentar si se quiere utilizar threads
	# frame = vs.read()


	# Descomentar si no se quiere utilizar threads
	_,frame=cap.read()
	# frame=cv2.imread("../../personas.jpg",cv2.COLOR_GRAY2BGR)
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	resized_image = cv2.resize(frame, (640, 480))

	blob = cv2.dnn.blobFromImage(frame, (1/127.5), (320, 240), 127.5, swapRB=True)

	net.setInput(blob)
	predictions = net.forward()

	centroids=[]
	boxes=[]

	persons=0

	for i in np.arange(0, predictions.shape[2]):
		confidence = predictions[0, 0, i, 2]
		if confidence > args["confidence"]:
			
			idx = int(predictions[0, 0, i, 1])


			if (CLASSES[idx] == 'person'):
				box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])

				(startX, startY, endX, endY) = box.astype("int")
				centerX = int(startX+(endX-startX)/2)
				centerY = int(startY+(endY-startY)/2)
				centroids.append((centerX,centerY))
				boxes.append((startX, startY, endX, endY))
				persons+=1

				# label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

	# ********* GUARDAR EN UN ARRAY NUMPY LOS DATOS DE TIEMPO Y FRAMES ***********
	
	v=set()

	if persons >=2:
		D=dist.cdist(centroids,centroids,metric="euclidean")
		for i in range(0,D.shape[0]):
			for j in range(i+1,D.shape[1]):
				print(D[i,j])
				if D[i,j] < 30:
					v.add(i)
					v.add(j)

	for i in range(len(boxes)):
		color=(0,255,0)
		if i in v:
			color=(0,0,255)
		cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]),color, 2)
		y = boxes[i][2] - 15 if boxes[i][2] - 15 > 15 else boxes[i][2] + 15
		cv2.circle(frame,(centroids[i][0],centroids[i][1]),1,(0,255,0),thickness=1)



	fps.stop()
	if first_time:
		measures[0][0]=abs(init_time-time.time())
		measures[0][1]=int(fps.fps())
		first_time=False
	else:
		measures.resize(len(measures)+1,NUM_DATA)
		measures[len(measures)-1][0]=abs(init_time-time.time())
		measures[len(measures)-1][1]=float(fps.fps())
	cv2.putText(frame, 'FPS: {:.2f}'.format(fps.fps()), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
	cv2.imshow("Frame", frame)
	fps = FPS().start()

	#********************************************************************************


	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	fps.update()

# np.save("rendimiento_con_threads",measures)
# np.save("rendimiento_sin_threads",measures)
fps.stop()

cv2.destroyAllWindows()
# Descomentar si se quiere utilizar threads
# vs.stop()
