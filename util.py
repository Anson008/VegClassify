import math
import ctypes
import os
import cv2
import numpy as np
import rioxarray as rxr

BETA = 1.1
WAYBACK_SCALE_TO_RESOLUTION = {"16": 0.9843, "18": 0.2237, "17": 0.4474}
TM_METHODS_STR = ('cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED')


class UnitConverterFactory:
    @staticmethod
    def create_converter(converter_type):
        try:
            if converter_type == "DegreeToRadian":
                return DegreeToRadian()
            elif converter_type == "RadianToDegree":
                return RadianToDegree()
            raise AssertionError("Converter type is not valid")
        except AssertionError as e:
            print(e)


class DegreeToRadian:
    @staticmethod
    def get_factor():
        return 2 * math.pi / 360.0


class RadianToDegree:
    @staticmethod
    def get_factor():
        return 360.0 / (2 * math.pi)


def get_screen_resolution():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    return width, height


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __str__(self):
        return f"Point ({self.x}, {self.y}"

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val


def create_grid(start, stop, num):
    return np.linspace(start, stop, num, endpoint=True)


def cv2_show_image(file_path, window_name, top_left=None, bottom_right=None, transpose=False):
    img = cv2.imread(file_path)
    if top_left is not None and bottom_right is not None:
        img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if transpose:
        img = cv2.transpose(img)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def get_wayback_shot_size(naip_size, naip_resolution, wayback_resolution):
    naip_h, naip_w = naip_size
    wayback_w = round(naip_h * naip_resolution / wayback_resolution * BETA)
    wayback_h = round(naip_w * naip_resolution / wayback_resolution * BETA)
    if wayback_w <= naip_w:
        wayback_w = round(naip_w * BETA)
    if wayback_h <= naip_h:
        wayback_h = round(naip_h * BETA)
    return wayback_h, wayback_w


def get_template_matching_scales(naip_resolution, wayback_resolution, n_points=4):
    # Alpha is the resizing factor that guarantees a Wayback imagery screenshot
    # having the same resolution as NAIP imagery
    alpha = naip_resolution / wayback_resolution
    delta = alpha * (BETA - 1)
    scales = np.linspace(alpha - delta, alpha + delta, n_points, endpoint=False)
    return scales


def get_ndvi_thresholds(start, stop, num):
    return np.linspace(start, stop, num, endpoint=True)


def get_naip_imagery_samples(naip_h, naip_w, n_samples_xy=(2, 2), seed=None):
    s_w = min(int(naip_w / n_samples_xy[0]), 512)
    s_h = min(int(naip_h / n_samples_xy[1]), 512)

    rng = np.random.default_rng(seed)
    top_left_x = rng.integers(0, naip_w - s_w, n_samples_xy[0], dtype=np.int32)
    top_left_y = rng.integers(0, naip_h - s_h, n_samples_xy[1], dtype=np.int32)

    bottom_right_x = top_left_x + s_w
    bottom_right_y = top_left_y + s_h

    top_left_xy = np.array(np.meshgrid(top_left_x, top_left_y)).T.reshape(-1, 2)
    bottom_right_xy = np.array(np.meshgrid(bottom_right_x, bottom_right_y)).T.reshape(-1, 2)

    diagonal_xy = np.concatenate((top_left_xy, bottom_right_xy), axis=1)
    # print(diagonal_xy)

    return diagonal_xy


def read_naip_image(image_path):
    naip = None
    try:
        naip = rxr.open_rasterio(image_path)
    except IOError as err:
        print(err)
    return naip


def create_directory(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as err:
            print(err)


def remove_all_files(directory):
    if len(os.listdir(directory)) == 0:
        return
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
        print(f"All cached NAIP crops are deleted at {directory}")
    except OSError as err:
        print(err)


def naip_to_bgr(naip_img):
    naip_np_arr = naip_img.values

    # Extract RGB channels and reorder the color axis from (c, w, h) to (w, h, c)
    naip_rgb = np.moveaxis(naip_np_arr[0:3], 0, -1)
    if naip_rgb is None:
        print("Naip image is empty")

    # Convert RGB to BGR, as BGR is the default color model of OpenCV
    return cv2.cvtColor(naip_rgb, cv2.COLOR_RGB2BGR)


def stitch_images(in_dir, out_dir):
    token = "rowSize"
    count = 0
    images = []
    row_size = 0
    try:
        with os.scandir(in_dir) as entries:
            sorted_entries = sorted(entries, key=lambda x: x.name)
        for entry in sorted_entries:
            # print(entry.name)
            if entry.is_file():
                if count == 0:
                    tokens = entry.name.split("_")
                    row_size = int(tokens[2][len(token):])
                    print(row_size)
                    count += 1
                images.append(cv2.imread(entry.path))
        img_rows = []
        for start_idx in range(0, len(images), row_size):
            # print(f"start_idx: {start_idx}")
            img_row = np.hstack(tuple(np.asarray(i) for i in images[start_idx:start_idx + row_size]))
            img_rows.append(img_row)
        imgs_merged = np.vstack(tuple(img_rows))
        create_directory(out_dir)
        cv2.imwrite(os.path.join(out_dir, f"merged_ground_truth.png"), imgs_merged)
    except OSError as err:
        print(err)


if __name__ == '__main__':
    in_dir = "./cache/naip_split/"
    out_dir = "./cache/naip_merged/"
    stitch_images(in_dir, out_dir)
