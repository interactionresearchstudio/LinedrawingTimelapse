#!/usr/bin/env python
import cv2
import imutils
import os
import numpy as np
import json
import time
import datetime
import subprocess
try:
    import RPi.GPIO as GPIO
    gpio_exists = True
except ImportError:
    gpio_exists = False
from threading import Thread
from CameraController import CameraController


class LinedrawingTimelapse(Thread):

    def __init__(self, configuration):
        self.cancelled = False

        self.config = configuration

        # Create full screen OpenCV window.
        if gpio_exists:
            cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, 1)

        # Start video stream.
        self.cam = CameraController.CameraController()
        self.cam.start()

        # Variable initialisation
        self.mode = 0
        self.lines = None
        self.avg = None
        self.canny_offset = 0
        self.is_recording = False
        self.last_capture_time = time.time()
        self.last_preview_time = time.time()
        self.last_countdown_time = time.time()
        self.capture_index = 0
        self.preview_index = 0
        self.first_capture = None
        self.countdown = None
        self.showing_live = False
        self.current_info_text = None
        self.live_start_time = time.time()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.key_pressed = None

        # Setup buttons
        self.btn1 = 17
        self.btn2 = 22
        self.btn3 = 23
        self.btn4 = 27
        self.btn_main = self.btn1
        if gpio_exists:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.btn1, GPIO.IN, GPIO.PUD_UP)
            GPIO.setup(self.btn2, GPIO.IN, GPIO.PUD_UP)
            GPIO.setup(self.btn3, GPIO.IN, GPIO.PUD_UP)
            GPIO.setup(self.btn4, GPIO.IN, GPIO.PUD_UP)

        self.anim_lenscap = ["assets/lensecap_white_1.png", "assets/lensecap_white_2.png"]
        self.anim_starting = ["assets/recording_in_1.png", "assets/recording_in_2.png", "assets/recording_in_3.png",
                              "assets/recording_in_4.png", "assets/recording_in_5.png"]
        self.graphic_default_lines = "assets/default_lines.png"
        self.graphic_less_lines = "assets/less_lines.png"
        self.graphic_more_lines = "assets/more_lines.png"
        self.graphic_live_preview = "assets/live_preview.png"

    def run(self):
        while not self.cancelled:
            new_frame = self.cam.get_image()
            if new_frame is not None:
                self.update(new_frame)
            else:
                print("Got None image.")

    def cancel(self):
        self.cam.stop()
        cv2.destroyAllWindows()
        self.cancelled = True

    def update(self, frame):

        if self.config["mirror_camera"] is 1:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.lines = self.auto_canny(gray, offset=self.canny_offset)

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

        self.key_pressed = cv2.waitKey(10)
        if self.key_pressed == 27:
            self.cancel()

    def standby(self, img):
        output = np.zeros((240, 320, 1), np.uint8)
        # output = self.insert_centered_text(output, "Open peephole to start recording.")

        current_anim_frame = int(time.time() % 2)
        output = self.paste_png(output, self.anim_lenscap[current_anim_frame])

        if self.config["flip_video"] is 1:
            output = imutils.rotate(output, angle=180)
        cv2.imshow("Output", output)

        if self.is_peephole_open(img) is True:
            self.mode = 1

    def starting(self, img):
        output = self.lines
        if self.countdown is None:
            # start countdown
            self.countdown = self.config["countdown"]
        elif self.countdown > 0:
            current_time = time.time()
            current_anim_frame = self.countdown - 1
            if output is not None and current_anim_frame < len(self.anim_starting) is not None:
                # display countdown
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                output = self.paste_png(output, self.anim_starting[current_anim_frame])
            if current_time - self.last_countdown_time >= 1:
                # iterate countdown clock
                self.countdown = self.countdown - 1
                self.last_countdown_time = current_time
        elif self.countdown is 0:
            # end countdown
            self.countdown = None
            self.mode = 2

        if self.config["flip_video"] is 1:
            output = imutils.rotate(output, angle=180)

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

        current_time = time.time()

        # capture frame
        if current_time - self.last_capture_time >= timelapse_frequency:
            file_name = "-%d.jpg" % self.capture_index
            if self.showing_live is False:
                cv2.imwrite(self.first_capture + file_name, self.lines)
                self.capture_index = self.capture_index + 1
            self.last_capture_time = current_time

        # display preview image
        if current_time - self.last_preview_time >= self.config["timelapse_preview_speed"] and self.capture_index > 1:
            if self.preview_index > self.capture_index-1:
                self.preview_index = 0

            current_file = self.first_capture + "-%d.jpg" % self.preview_index
            current_frame = cv2.imread(current_file, cv2.IMREAD_COLOR)
            if current_frame is not None:
                if self.config["flip_video"] is 1:
                    current_frame = imutils.rotate(current_frame, angle=180)

                if self.showing_live is False:
                    cv2.imshow("Output", current_frame)

            self.preview_index = self.preview_index + 1
            self.last_preview_time = current_time

        # show / hide live image
        if gpio_exists:
            if GPIO.input(self.btn_main) is False:
                self.showing_live = not self.showing_live
                self.live_start_time = current_time
                time.sleep(0.2)
        else:
            if self.key_pressed == ord('1'):
                self.showing_live = not self.showing_live
                self.live_start_time = current_time
                time.sleep(0.2)

        # live mode
        if self.showing_live is True:
            live_frame = self.lines

            # display live image
            if self.lines is not None:
                live_frame = cv2.cvtColor(live_frame, cv2.COLOR_GRAY2BGR)

                # show info text
                if current_time - self.live_start_time <= self.config["info_text_timeout"]:
                    if self.current_info_text is not None:
                        live_frame = self.paste_png(live_frame, self.current_info_text)
                else:
                    self.current_info_text = None
                    live_frame = self.paste_png(live_frame, self.graphic_live_preview)

                # show live image
                if self.config["flip_video"] is 1:
                    live_frame = imutils.rotate(live_frame, angle=180)
                cv2.imshow("Output", live_frame)

            # adjust auto canny
            if gpio_exists:
                if GPIO.input(self.btn2) is False:
                    self.canny_offset = self.canny_offset + self.config["canny_offset_step"]
                    self.current_info_text = self.graphic_less_lines
                    self.live_start_time = time.time()
                    time.sleep(0.1)
                elif GPIO.input(self.btn4) is False:
                    self.canny_offset = self.canny_offset - self.config["canny_offset_step"]
                    self.current_info_text = self.graphic_more_lines
                    self.live_start_time = time.time()
                    time.sleep(0.1)
                elif GPIO.input(self.btn3) is False:
                    self.canny_offset = 0
                    self.current_info_text = self.graphic_default_lines
                    self.live_start_time = time.time()
                    time.sleep(0.1)
            else:
                if self.key_pressed == ord('2'):
                    self.canny_offset = self.canny_offset + self.config["canny_offset_step"]
                    self.current_info_text = self.graphic_less_lines
                    self.live_start_time = time.time()
                    time.sleep(0.1)
                if self.key_pressed == ord('4'):
                    self.canny_offset = self.canny_offset - self.config["canny_offset_step"]
                    self.current_info_text = self.graphic_more_lines
                    self.live_start_time = time.time()
                    time.sleep(0.1)
                if self.key_pressed == ord('3'):
                    self.canny_offset = 0
                    self.current_info_text = self.graphic_default_lines
                    self.live_start_time = time.time()
                    time.sleep(0.1)

            # timeout live mode
            if current_time - self.live_start_time >= self.config["live_timeout"]:
                self.showing_live = False

        if self.is_peephole_open(img) is False:
            if gpio_exists:
                self.save_mp4(self.first_capture)
            self.showing_live = False
            self.canny_offset = 0
            self.first_capture = None
            self.mode = 0

    def is_peephole_open(self, g):
        fixed_lines = cv2.Canny(g, self.config["canny_threshold"], self.config["canny_ratio"] *
                                self.config["canny_threshold"], apertureSize=self.config["canny_aperturesize"])
        non_zeros = cv2.countNonZero(fixed_lines)

        if non_zeros <= self.config["peephole_threshold"]:
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
        x_pos = (w - textsize[0]) / 2
        y_pos = (h - textsize[1]) / 2
        if on_rectangle is True:
            cv2.rectangle(img, (x_pos - 1, y_pos - textsize[1]), (x_pos + textsize[0] + 1, y_pos + 1), 0, -1)
        cv2.putText(img, txt, (x_pos, y_pos), self.font, size, 255, stroke, cv2.LINE_AA)
        return img

    @staticmethod
    def save_mp4(f):
        subprocess.Popen(["avconv", "-r", "2", "-start_number", "1", "-i", f + "-%d.jpg", "-b:v",
                          "1000k", f + ".mp4"])

    def auto_canny(self, image, offset=0, sigma=0.33):
        v = np.median(image) + offset
        v = self.constrain(v, 0, 255)

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        return edged

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
    if gpio_exists:
        os.chdir("/home/pi/LinedrawingTimelapse")
    config = json.load(open("conf.json"))

    print("[INFO] Started main.")

    timelapse_instance = LinedrawingTimelapse(config)
    timelapse_instance.run()


if __name__ == '__main__':
    main()
