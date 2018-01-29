# For notes to self cmd+f for "NOTE:"
#Import packages
import argparse
import datetime
import signal
import sys
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-o", "--output", help="path to the output folder")
ap.add_argument("-a", "--min-area", type=int, default=800, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture("rtsp://192.168.5.109:554/s1") #TODO: make this a command line argument
	time.sleep(1.0)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# location to save output must be specified
if args.get("output", None) is None:
	print('You must specify output folder for saved images with -a argument')
	sys.exit(1)
else:
	outputFolder = args["output"]
	
referenceFrame = None  
frameCount = 0
motionPresent = False
updateFrequency = 90

#Define what happens when ctrl+c is pressed
def signal_handler(signal, frame):
	camera.release()
	print("\nExiting")
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	frameCount += 1
	motionPresent = False

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	hires = frame
	frame = imutils.resize(frame, width=480)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0) # NOTE: more blur may be needed for outdoors with wind 

	# if the first frame is None, initialize it
	if referenceFrame is None:
		referenceFrame = gray
		continue
    
    # compute the absolute difference between the current frame and first frame
	frameDelta = cv2.absdiff(referenceFrame, gray)
	thresh = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_BINARY)[1] # NOTE: 2nd arg is difference threshhold.
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=3)
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = list(filter(lambda x: cv2.contourArea(x) > args["min_area"], cnts)) # Filter the list of contours to just those greater than the minumum area
	if cnts:
		motionPresent = True

	if motionPresent:
		# Find the bounding box that contains all of the significant contours
		minX = (min(list(map(lambda x: cv2.boundingRect(x)[0], cnts))))
		minY = (min(list(map(lambda x: cv2.boundingRect(x)[1], cnts))))
		maxW = (max(list(map(lambda x: minX + cv2.boundingRect(x)[2], cnts))))
		maxH = (max(list(map(lambda x: minY + cv2.boundingRect(x)[3], cnts))))

		crop = hires[minY * 4:maxH * 4, minX * 4:maxW * 4] #TODO: get rid of these magic numbers
		cv2.imwrite(outputFolder + str(frameCount) + '.jpg', crop, [])

	# update reference frame
	if motionPresent == False and frameCount % updateFrequency == 0:
		referenceFrame = gray
