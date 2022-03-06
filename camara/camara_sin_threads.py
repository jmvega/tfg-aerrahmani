from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
from threading import Thread


NUM_DATA=2
measures = np.zeros([1,NUM_DATA])
first_time=True





cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
fps = FPS().start()
init_time=time.time()
index=0
counter=0
fps_avg_frame_count = 10
start_time = time.time()
sec=time.time()

while True:
	_,frame=cap.read()
	
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	
	resized_image = cv2.resize(frame, (640, 480))

	fps.stop()
	if(abs(time.time()-sec)>=0.5):
		if first_time:
			measures[0][0]=abs(init_time-time.time())
			measures[0][1]=fps.fps()
			first_time=False
		else:
			measures.resize(len(measures)+1,NUM_DATA)
			measures[len(measures)-1][0]=abs(init_time-time.time())
			measures[len(measures)-1][1]=float(fps.fps())
		sec=time.time()

    
	fps_text = 'FPS = {:.1f}'.format(fps.fps())

	cv2.putText(frame, fps_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
	cv2.imshow('object detection',  frame)


	key = cv2.waitKey(1) & 0xFF
	fps=FPS().start()
	if key == ord("q"):
		break

	fps.update()

np.save("rendimiento_sin_threads",measures)
fps.stop()

cv2.destroyAllWindows()


