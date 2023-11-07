import numpy as np
import cv2
from connected_components import ConnectedComponentsProcessor as ccp


class ColorDetector:
    def __init__(self, path=None):
        if path:
            self._img = cv2.imread(path)
        else:
            self._img = None

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, path):
        self._img = cv2.imread(path)

    def apply_hist_equalization(self, channel=2):
        hsv_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
        hsv_img[:, :, channel] = cv2.equalizeHist(hsv_img[:, :, channel])
        return hsv_img, cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    def detect_color(self, lower_hue=40, upper_hue=80):
        lower = np.array([lower_hue, 30, 30])
        upper = np.array([upper_hue, 255, 255])
        hsv_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_img, lower, upper)


if __name__ == "__main__":
    detector = ColorDetector("./image/naip_rgb.png")
    mask = detector.detect_color()
    bgr_img = detector.img

    res = ccp.overlap_on_map(mask, bgr_img, "red")
    # cv2.imwrite('./results/color_detection_H40-80_S30-255_V30-255.png', res)

