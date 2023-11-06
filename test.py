import rioxarray as rxr
import earthpy.plot as ep
import matplotlib.pyplot as plt
import numpy as np
import math
import tkinter
import ctypes
import cv2
from connected_components import ConnectedComponentsProcessor as ccp


img_path = "./image/naip_rgb.png"
bgr_img = cv2.imread(img_path)
print("Shape of image:", bgr_img.shape)

cv2.imshow("RGB", bgr_img)
cv2.waitKey(0)

hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv_img)
cv2.waitKey(0)

# color_space = np.max(hsv_img[190:240, 430:, 2])
# print(color_space)

hue_green = 60
sensitivity = 20

# print(hsv_img[190:240, 430:, 0])
lower = np.array([40, 30, 30])
upper = np.array([80, 255, 255])
mask = cv2.inRange(hsv_img, lower, upper)
# hue = (hsv_img[:, :, 0] > hue_green - sensitivity) & (hsv_img[:, :, 0] < hue_green + sensitivity)
# print(mask)
# print(hue[190:240, 430:])

# mask = hue.astype("uint8") * 255
res = ccp.overlap_on_map(mask, bgr_img, "red")
cv2.imshow("Result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()


