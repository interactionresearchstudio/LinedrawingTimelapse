#!/usr/bin/env python
#define pi

import cv2
import imutils
from imutils.video.pivideostream import PiVideoStream
import json
import time
import os
from threading import Thread


class LinedrawingTimelapse(Thread):

    def __init__(self, configuration):
        super().__init__()
        self.cancelled = False

        self.config = configuration

        # Start video stream.
        self.vs = PiVideoStream().start()
        time.sleep(self.config["camera_warmup_time"])

        # Create full screen OpenCV window.
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, 1)

        self.mode = 0
        self.lines = None
        self.avg = None
        self.is_recording = False

    def run(self):
        while not self.cancelled:
            self.update()

    def cancel(self):
        self.vs.stop()
        cv2.destroyAllWindows()
        self.cancelled = True

    def update(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=320, height=240)

        if self.config["flip_video"] is 1:
            frame = imutils.rotate(frame, angle=180)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.lines = imutils.auto_canny(gray)

        cv2.imshow("Output", self.lines)

    def detect_timelapse_frequency(self, img):
        img = cv2.GaussianBlur(img, (21, 21), 0)

        if self.avg is None:
            self.avg = img.copy().astype("float")

        cv2.accumulateWeighted(img, self.avg, self.config["delta_threshold"])
        frame_delta = cv2.absdiff(img, cv2.convertScaleAbs(self.avg))
        motion_factor = 1 - ((cv2.countNonZero(frame_delta) + 0.0) / (frame_delta.shape[0] * frame_delta.shape[1]))
        motion_factor = self.constrain(motion_factor, self.config["min_motion_factor"],
                                       self.config["max_motion_factor"])

        return self.map_factor(motion_factor, self.config["min_motion_factor"], self.config["max_motion_factor"],
                          self.config["min_timelapse_frequency"], self.config["max_timelapse_frequency"])

    def record(self, img):
        print("Recording...")
        # record some stuff.
        # check timelapse frequency
        # take photo if it's time.
        # loop through photos.

    def preview(self, img):
        print("Previewing...")
        # preview the image before the

    @staticmethod
    def constrain(val, min_val, max_val):
        return max(min(max_val, val), min_val)

    @staticmethod
    def map_factor(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# load configuration file
os.chdir("/home/pi/LinedrawingTimelapse")
conf = json.load(open("conf.json"))

def main():
    os.chdir("/home/pi/LinedrawingTimelapse")
    config = json.load(open("config.json"))
    timelapseInstance = LinedrawingTimelapse(config)


if __name__ == '__main__':
    main()

