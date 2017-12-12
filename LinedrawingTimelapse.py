#!/usr/bin/env python

import cv2
import imutils
from imutils.video.pivideostream import PiVideoStream
import numpy as np
import json
import time
import datetime
import os
import subprocess
from threading import Thread


class LinedrawingTimelapse(Thread):

    def __init__(self, configuration):
        self.cancelled = False

        self.config = configuration

        # Create full screen OpenCV window.
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, 1)

        # Start video stream.
        self.vs = PiVideoStream().start()
        time.sleep(self.config["camera_warmup_time"])

        self.mode = 0
        self.lines = None
        self.avg = None
        self.is_recording = False
        self.last_capture_time = time.time()
        self.last_preview_time = time.time()
        self.last_countdown_time = time.time()
        self.capture_index = 0
        self.preview_index = 0
        self.first_capture = None
        self.countdown = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

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

        if self.config["mirror_camera"] is 1:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.lines = imutils.auto_canny(gray)

        if self.mode is 0:
            self.standby(gray)

        elif self.mode is 1:
            self.starting(gray)
            if self.is_peephole_open(gray) is False:
                self.mode = 0

        elif self.mode is 2:
            self.recording(gray)
            if self.is_peephole_open(gray) is False:
                self.mode = 0

        cv2.waitKey(10)


    def standby(self, img):
        output = np.zeros((240, 320, 1), np.uint8)
        output = self.insert_centered_text(output, "Open peephole to start recording.")
        cv2.imshow("Output", output)

        if self.is_peephole_open(img) is True:
            self.mode = 1

    def starting(self, img):
        output = self.lines
        # preview the image before the
        if self.countdown is None:
            self.countdown = self.config["countdown"]
        elif self.countdown > 0:
            output = self.insert_centered_text(output, "Starting in " + str(self.countdown), on_rectangle=True)
            current_time = time.time()
            if current_time - self.last_countdown_time >= 1:
                self.countdown = self.countdown - 1
                self.last_countdown_time = current_time
        elif self.countdown is 0:
            self.countdown = None
            self.mode = 2

        cv2.imshow("Output", output)

        if self.is_peephole_open(img) is False:
            self.countdown = None
            self.mode = 0


    def recording(self, img):
        # get current timelapse frequency
        timelapse_frequency = self.get_timelapse_frequency(img)

        # define first capture file name
        if self.first_capture is None:
            timestamp = datetime.datetime.now()
            self.first_capture = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

        # capture frame
        current_time = time.time()
        if current_time - self.last_capture_time >= timelapse_frequency:
            file_name = "-%d.jpg" % self.capture_index
            cv2.imwrite(self.first_capture + file_name, self.lines)
            self.capture_index = self.capture_index + 1
            self.last_capture_time = current_time

        # display preview image
        if current_time - self.last_preview_time >= self.config["timelapse_preview_speed"] and self.capture_index > 1:
            if self.preview_index > self.capture_index-1:
                self.preview_index = 0

            current_file = self.first_capture + "-%d.jpg" % self.preview_index
            current_frame = cv2.imread(current_file, cv2.IMREAD_COLOR)
            cv2.imshow("Output", current_frame)
            self.preview_index = self.preview_index + 1
            self.last_preview_time = current_time

        if self.is_peephole_open(img) is False:
            self.save_mp4(self.first_capture)
            self.first_capture = None
            self.mode = 0

    def is_peephole_open(self, g):
        fixed_lines = cv2.Canny(g, self.config["canny_threshold"], self.config["canny_ratio"] * self.config["canny_threshold"],
                          apertureSize=self.config["canny_aperturesize"])
        non_zeros = cv2.countNonZero(fixed_lines)

        if non_zeros == 0:
            return False
        else:
            return True

    def get_timelapse_frequency(self, img):
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

    def insert_centered_text(self, img, txt, on_rectangle=False, size=0.5, stroke=1):
        textsize, _ = cv2.getTextSize(txt, self.font, size, stroke)
        h, w = img.shape[:2]
        xPos = (w - textsize[0]) / 2
        yPos = (h - textsize[1]) / 2
        if on_rectangle is True:
            cv2.rectangle(img, (xPos - 1, yPos - textsize[1]), (xPos + textsize[0] + 1, yPos + 1), (0), -1)
        cv2.putText(img, txt, (xPos, yPos), self.font, size, (255), stroke, cv2.LINE_AA)
        return img

    def save_mp4(self, f):
        subprocess.Popen(["avconv", "-r", "2", "-start_number", "1", "-i", f + "-%d.jpg", "-b:v",
                          "1000k", f + ".mp4"])

    @staticmethod
    def paste_png(base, overlay_filename):
        overlay = cv2.imread(overlay_filename, -1)
        overlay_bgr = overlay[:, :, :3]
        overlay_mask = overlay[:, :, 3:]

        # inverse mask
        bg_mask = 255 - overlay_mask

        # turn masks into three channel masks
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

        base_part = (base * (1 / 255.0)) * (bg_mask * (1 / 255.0))
        overlay_part = (overlay_bgr * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        return np.uint8(cv2.addWeighted(base_part, 255.0, overlay_part, 255.0, 0.0))

    @staticmethod
    def constrain(val, min_val, max_val):
        return max(min(max_val, val), min_val)

    @staticmethod
    def map_factor(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def main():
    os.chdir("/home/pi/LinedrawingTimelapse")
    config = json.load(open("conf.json"))

    print("[INFO] Started main.")

    timelapseInstance = LinedrawingTimelapse(config)
    timelapseInstance.run()

if __name__ == '__main__':
    main()

