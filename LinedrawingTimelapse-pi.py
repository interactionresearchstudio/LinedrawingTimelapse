#!/usr/bin/env python
#define pi

import cv2
import json
import time
import datetime
import numpy as np
import os

from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

# load configuration file
os.chdir("/home/pi/LinedrawingTimelapse")
conf = json.load(open("conf.json"))

cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, 1)

camera = PiCamera()
camera.resolution=(320,240)
camera.framerate=32
rawCapture = PiRGBArray(camera, size=(320,240))

time.sleep(0.1)

# buttons
btn1 = 17
btn2 = 22
btn3 = 23
btn4 = 27
btnShutter = btn1
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(btn1, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btn2, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btn3, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btn4, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btnShutter, GPIO.IN, GPIO.PUD_UP)

avg = None

previousPictureTime = 0
previousCheckTime = 0
previousTimelapseChange = 0
pictureFrequency = 1
previousMotionFactor = 0.0
startPicture = ""
imageIndex = 0
timelapseIndex = 0
numOfPhotos = 0

# modes:
# 0 - Standby
# 1 - Recording
# 2 - Stopped
mode = 0

font = cv2.FONT_HERSHEY_SIMPLEX

# map
def mapFactor(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# start rec
def startRecording():
    global mode
    global previousPictureTime
    global previousCheckTime
    global pictureFrequency
    global previousMotionFactor
    global startPicture
    global imageIndex
    global numOfPhotos
    global previousTimelapseChange
    global timelapseIndex
    imageIndex = 0
    numOfPhotos = 0
    previousPictureTime = 0
    previousCheckTime = 0
    pictureFrequency = 1
    previousMotionFactor = 0.0
    previousTimelapseChange = 0
    timelapseIndex = 0
    timestamp = datetime.datetime.now()
    startPicture = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    print("[INFO] Start picture: " + startPicture)
    mode = 1
    print("[INFO] Started recording")
    return

def stopRecording():
    global numOfPhotos
    numOfPhotos = imageIndex
    print("Finished with a total of %d photos" % numOfPhotos)
    compileTimelapse(startPicture)

def showTimelapse():
    global mode
    global imageIndex
    mode = 2
    imageIndex = 0

def rotateImage(img):
    (h,w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    return cv2.warpAffine(img, M, (w,h))

def compileTimelapse(filename):
    command = "gst-launch-1.0 multifilesrc location=" + filename + "-%d.jpg index=1 caps='image/jpeg,framerate=4/1' ! jpegdec ! omxh264enc ! avimux ! filesink location=" + filename + ".avi"
    os.system(command)

def convertToLineImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, conf["canny_threshold"],conf["canny_ratio"]*conf["canny_threshold"], apertureSize = conf["canny_aperturesize"])
    return edges

def insertCentredText(img, txt):
    textsize, _ = cv2.getTextSize(txt, font, 0.5, 1)
    h, w = img.shape[:2]
    xPos = (w - textsize[0]) / 2
    yPos = (h - textsize[1]) / 2
    cv2.putText(img, txt, (xPos, yPos), font, 0.5, (255), 1, cv2.LINE_AA)
    return img

# main cv loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image = cv2.flip(image, 1)
    image = cv2.flip(image, 0)
    
    if mode is 0:
        # standby.
        if GPIO.input(btnShutter) == False:
            # start recording. 
            startRecording()
        if conf["flip_camera"] is 1:
            image = rotateImage(image)
        lines = convertToLineImage(image)
        nonZeros = cv2.countNonZero(lines)
        if nonZeros == 0:
            insertCentredText(lines, "Open peephole\r\nto start recording")
        cv2.imshow("Output", lines)
        
    if mode is 1:
        # grab frame, resize and convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

        # capture first frame of background model
        if avg is None:
            print "[INFO] First frame of background model."
            avg = gray.copy().astype("float")
    
        # accumulate new frame
        cv2.accumulateWeighted(gray, avg, conf["delta_threshold"])
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        motionFactor = 1 - ((cv2.countNonZero(frameDelta) + 0.0) / (frameDelta.shape[0] * frameDelta.shape[1]))
        #print("[CALC] Motion factor: %f") % motionFactor
        # keep motion factor within limits
        if motionFactor < conf["min_motion_factor"]:
            motionFactor = conf["min_motion_factor"]
        if motionFactor > conf["max_motion_factor"]:
            motionFactor = conf["max_motion_factor"]

        # change picture frequency
        pictureFrequency = mapFactor(motionFactor, conf["min_motion_factor"], conf["max_motion_factor"], conf["min_timelapse_frequency"], conf["max_timelapse_frequency"])

        #print("[CALC] Picture frequency: %f") % pictureFrequency

        currentTime = time.time()
        if currentTime - previousPictureTime > pictureFrequency:
            fileName = "-%d.jpg" % imageIndex
            if conf["flip_video"] is 1:
                image = rotateImage(image)
            lines = convertToLineImage(image)
            cv2.imwrite(startPicture + fileName, lines)
            imageIndex = imageIndex + 1
            previousPictureTime = currentTime
            print("[INFO] Picture saved.")

        # show delta
        #cv2.imshow("Output", frameDelta)
        if currentTime - previousTimelapseChange >= conf["timelapse_preview_speed"] and imageIndex > 1:
            if timelapseIndex > imageIndex-1:
                timelapseIndex = 0
            currentFileName = startPicture + "-%d.jpg" % timelapseIndex
            currentFrame = cv2.imread(currentFileName, cv2.IMREAD_COLOR)
            if conf["flip_camera"] is 1 or conf["flip_video"] is 1:
                currentFrame = rotateImage(currentFrame)
            cv2.imshow("Output", currentFrame) 
            print("[INFO] Showing picture %d" % timelapseIndex)
            timelapseIndex = timelapseIndex + 1
            previousTimelapseChange = currentTime
        
        if GPIO.input(btnShutter) == False and imageIndex > 2:
            # start recording. 
            stopRecording()
            #showTimelapse()
            mode = 0
            time.sleep(0.5)

    if mode is 2:
        currentTime = time.time()
        # show previous timelapse
        if currentTime - previousPictureTime >= conf["timelapse_preview_speed"]:
            if imageIndex > numOfPhotos - 1:
                imageIndex = 0
            currentFileName = startPicture + "-%d.jpg" % imageIndex
            currentFrame = cv2.imread(currentFileName, cv2.IMREAD_COLOR)
            if conf["flip_camera"] is 1:
                currentFrame = rotateImage(currentFrame)
            cv2.imshow("Output", currentFrame)
            print("[INFO] Showing picture %d" % imageIndex)
            imageIndex = imageIndex + 1
            previousPictureTime = currentTime

        if GPIO.input(btnShutter) == False:
            # back to standby
            mode = 0
            time.sleep(0.5)

    # keys
    key = cv2.waitKey(10)
    rawCapture.truncate(0)
    if key == ord("0"):
        mode = 0
    if key == ord("1"):
        startRecording()
    if key == ord("2"):
        stopRecording()
        showTimelapse()
    if key == 27:
        break

cv2.destroyWindow("preview")
