import math
import ctypes
import os
import cv2
import json
import numpy as np
import rioxarray as rxr

# Constants
BETA = 1.1
WAYBACK_SCALE_TO_RESOLUTION = {"16": 0.9843, "18": 0.2237, "17": 0.4474}
TM_METHODS_STR = ('cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED')
NAIP_SPLIT_DIR = "./cache/naip_split/"
NAIP_RANDOM_SAMPLES_DIR = "./cache/naip_random_samples/"
WAYBACK_SCREENSHOTS_DIR = "./cache/wayback_screenshots/"
GROUND_TRUTH_MASKS_DIR = "./cache/ground_truth_masks/"
GROUND_TRUTH_IMAGES_DIR = "./cache/ground_truth_images/"
# SELENIUM_DRIVER_BINARY_LOC = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
SELENIUM_DRIVER_BINARY_LOC = "C:/Program Files/ChromeDriver/chromedriver.exe"
NAIP_SAMPLE_MASKS_DIR = "./cache/naip_random_sample_masks/"
NAIP_BEST_MATCH_DIR = "./cache/naip_best_match/"
GROUND_TRUTH_MASKS_BINARY_DIR = "./cache/ground_truth_masks_binary/"
NAIP_MASKS_BINARY_DIR = "./cache/naip_masks_binary/"
TM_MATCH_INFO = "./cache/tm_match_info/"
TEMPLATE_MATCH_REGION = "./cache/template_match_region/"
BEST_MATCH_IMG_DIR = "./cache/best_match_img/"


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


def get_wayback_shot_size(naip_size, naip_resolution, wayback_resolution, beta=BETA):
    naip_h, naip_w = naip_size
    alpha = naip_resolution / wayback_resolution
    wayback_h = round(naip_h * alpha * beta)
    wayback_w = round(naip_w * alpha * beta)
    if wayback_w <= naip_w:
        wayback_w = round(naip_w * beta)
    if wayback_h <= naip_h:
        wayback_h = round(naip_h * beta)
    return wayback_h, wayback_w


def get_template_matching_scales(naip_resolution, wayback_resolution, n_points=4, beta=BETA):
    # Alpha is the resizing factor that guarantees a Wayback imagery screenshot
    # having the same resolution as NAIP imagery
    alpha = naip_resolution / wayback_resolution
    delta = alpha * (beta - 1)
    scales = np.linspace(alpha - delta, alpha + delta, n_points, endpoint=False)
    return scales


def get_ndvi_thresholds(start, stop, num):
    return np.linspace(start, stop, num, endpoint=True)


def get_random_naip_imagery_samples(naip_h, naip_w, n_samples_xy=(2, 2), seed=None):
    s_w = min(int(naip_w / n_samples_xy[0]), 256)
    s_h = min(int(naip_h / n_samples_xy[1]), 512)

    rng = np.random.default_rng(seed)
    top_left_x = rng.integers(0, naip_w - s_w, n_samples_xy[0], dtype=np.int32)
    top_left_y = rng.integers(0, naip_h - s_h, n_samples_xy[1], dtype=np.int32)

    bottom_right_x = top_left_x + s_w
    bottom_right_y = top_left_y + s_h

    return make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def get_grid_center(naip_h, naip_w):
    block_h = 512 if naip_h >= 512 else naip_h
    block_w = 256 if naip_w >= 256 else naip_w

    bottom_right_x = np.arange(block_w, naip_w, block_w, dtype=np.int32)
    bottom_right_y = np.arange(block_h, naip_h, block_h, dtype=np.int32)
    top_left_x = bottom_right_x - block_w
    top_left_y = bottom_right_y - block_h

    return make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    top_left_xy = np.array(np.meshgrid(top_left_x, top_left_y)).T.reshape(-1, 2)
    bottom_right_xy = np.array(np.meshgrid(bottom_right_x, bottom_right_y)).T.reshape(-1, 2)
    diagonal_xy = np.concatenate((top_left_xy, bottom_right_xy), axis=1)
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


def get_split_row_size():
    token = "rowSize"
    try:
        with os.scandir(NAIP_SPLIT_DIR) as entries:
            for entry in entries:
                if entry.is_file():
                    tokens = entry.name.split("_")
                    row_size = int(tokens[2][len(token):])
                    return row_size
    except OSError as err:
        print(err)


def stitch_images(in_dir, out_dir):
    try:
        with os.scandir(in_dir) as entries:
            sorted_entries = sorted(entries, key=lambda x: x.name)
    except OSError as err:
        print(err)

    images = []
    for entry in sorted_entries:
        if entry.is_file():
            images.append(cv2.imread(entry.path))

    row_size = get_split_row_size()
    img_rows = []
    for start_idx in range(0, len(images), row_size):
        img_row = np.hstack(tuple(np.asarray(i) for i in images[start_idx:start_idx + row_size]))
        img_rows.append(img_row)
    imgs_merged = np.vstack(tuple(img_rows))
    create_directory(out_dir)
    cv2.imwrite(os.path.join(out_dir, f"merged_ground_truth.png"), imgs_merged)


def get_image_center(top_left, bottom_right):
    tx, ty = top_left
    bx, by = bottom_right
    return (bx - tx) // 2, (by - ty) // 2


def match_template(ground_truth_mask, naip_mask, naip_resolution, wayback_resolution):
    tm_scales = get_template_matching_scales(naip_resolution, wayback_resolution)
    global_max = float('-inf')
    optimal_metrics = tuple()
    for i, scale in enumerate(tm_scales):
        ndvi_best = cv2.resize(naip_mask.copy(),
                               dsize=(0, 0),
                               fx=scale,
                               fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        similarity_res = cv2.matchTemplate(ground_truth_mask, ndvi_best, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity_res)
        if max_val > global_max:
            global_max = max_val
            optimal_metrics = (max_val, max_loc, ndvi_best)

    return optimal_metrics


def draw_template_match_region(wayback_shot_path, tm_info_path):
    wayback_shot_file_obj = os.scandir(wayback_shot_path)
    tm_info = np.load(tm_info_path)
    # print(f"tm_info shape: {tm_info.shape}")
    tm_info = tm_info.tolist()

    for i, item in enumerate(zip(tm_info, wayback_shot_file_obj)):
        tm_param, wayback_shot_file = item
        filename = wayback_shot_file.name
        if filename.endswith(".png"):
            wayback_img = cv2.imread(os.path.join(wayback_shot_path, filename))
            # print(tm_param)
            _, ox, oy, h, w = tm_param
            cv2.rectangle(wayback_img, (int(ox), int(oy)), (int(ox + w), int(oy + h)), (0, 255, 255), 3)
            out_filename = f"template_match_region_{i + 1}.png"
            cv2.imwrite(os.path.join(TEMPLATE_MATCH_REGION, out_filename), wayback_img)


def get_confusion_matrix(naip_mask_path, ground_truth_mask_path, tm_info_path):
    naip_file_obj = os.scandir(naip_mask_path)
    gt_file_obj = os.scandir(ground_truth_mask_path)
    tm_info = np.load(tm_info_path).tolist()
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "kappa": 0}
    tp, fp, tn, fn, kappa = 0, 0, 0, 0, 0

    for naip_mask, gt_mask, tm_i in zip(naip_file_obj, gt_file_obj, tm_info):
        scale, ox, oy, ndvi_h, ndvi_w = tm_i
        ox = int(ox)
        oy = int(oy)
        ndvi_h = int(ndvi_h)
        ndvi_w = int(ndvi_w)

        naip_mask_img = cv2.imread(os.path.join(naip_mask_path, naip_mask.name))
        gt_mask_img = cv2.imread(os.path.join(ground_truth_mask_path, gt_mask.name))

        gt_matched_mask = gt_mask_img[oy:oy + ndvi_h, ox:ox + ndvi_w]
        naip_matched_mask = cv2.resize(naip_mask_img,
                                       dsize=(0, 0),
                                       fx=scale,
                                       fy=scale,
                                       interpolation=cv2.INTER_CUBIC)

        # Accumulate TP and TN
        gt_and_naip = np.logical_and(gt_matched_mask, naip_matched_mask)
        n_tp = np.sum(gt_and_naip).astype(np.int64)
        tp += n_tp
        tn += gt_and_naip.size - n_tp

        # Accumulate FP and FN
        fp += np.sum(np.logical_and(np.logical_not(gt_matched_mask), naip_matched_mask)).astype(np.int64)
        fn += np.sum(np.logical_and(gt_matched_mask, np.logical_not(naip_matched_mask))).astype(np.int64)

    kappa = 2.0 * (tp * tn - fp * fn) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))

    cm["tp"] = int(tp)
    cm["fp"] = int(fp)
    cm["tn"] = int(tn)
    cm["fn"] = int(fn)
    cm["kappa"] = float(kappa)

    with open("./cache/confusion_matrix.json", "w+") as outfile:
        outfile.write(json.dumps(cm, indent=4))
    return cm


if __name__ == '__main__':
    # in_dir = "./cache/naip_split/"
    # out_dir = "./cache/naip_merged/"
    # stitch_images(in_dir, out_dir)

    samples = get_random_naip_imagery_samples(512, 1024, (2, 4), 101)
    print(samples)
    print(samples.shape)


