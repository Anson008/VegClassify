import math
import ctypes
import os
import cv2
import numpy as np
import rioxarray as rxr
# import constants as const
from utility.confusion_matrix import ConfusionMatrix

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
TEMPLATE_MATCH_REGION_DIR = "./cache/template_match_region/"
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


def read_naip_image(image_path):
    naip = None
    try:
        naip = rxr.open_rasterio(image_path)
    except IOError as err:
        print(err)
    return naip


def is_directory_exists(directory):
    return os.path.exists(directory)


def create_directory(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except OSError as err:
            print(err)
            return False
    else:
        print(f"Directory already exists: {directory}")
        return False


def remove_all_files(directory):
    if len(os.listdir(directory)) == 0:
        print("Directory is empty.")
        return True
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
        return True
    except OSError as err:
        print(err)
        return False


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
            cv2.imwrite(os.path.join(TEMPLATE_MATCH_REGION_DIR, out_filename), wayback_img)


def get_confusion_matrix_on_naip(naip_mask_path, ground_truth_mask_path):
    naip_file_obj = os.scandir(naip_mask_path)
    gt_file_obj = os.scandir(ground_truth_mask_path)
    cm_obj = ConfusionMatrix()

    for naip_mask, gt_mask in zip(naip_file_obj, gt_file_obj):

        naip_mask_img = cv2.imread(os.path.join(naip_mask_path, naip_mask.name))
        gt_mask_img = cv2.imread(os.path.join(ground_truth_mask_path, gt_mask.name))

        # Accumulate TP and TN
        gt_and_naip = np.logical_and(gt_mask_img, naip_mask_img)
        n_tp = np.sum(gt_and_naip).astype(np.int64)
        cm_obj.tp += n_tp
        cm_obj.tn += gt_and_naip.size - n_tp

        # Accumulate FP and FN
        cm_obj.fp += np.sum(np.logical_and(np.logical_not(gt_mask_img), naip_mask_img)).astype(np.int64)
        cm_obj.fn += np.sum(np.logical_and(gt_mask_img, np.logical_not(naip_mask_img))).astype(np.int64)

    # kappa = 2.0 * (tp * tn - fp * fn) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    # accuracy = 1.0 * (tp + tn) / (tp + fp + tn + fn)
    cm_obj.confusion_matrix["kappa"] = cm_obj.get_kappa()
    cm_obj.confusion_matrix["accuracy"] = cm_obj.get_accuracy()

    # cm["tp"] = int(tp)
    # cm["fp"] = int(fp)
    # cm["tn"] = int(tn)
    # cm["fn"] = int(fn)
    # cm["kappa"] = float(kappa)
    # cm["accuracy"] = float(accuracy)

    return cm_obj.confusion_matrix


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
    accuracy = 1.0 * (tp + tn) / (tp + fp + tn + fn)

    cm["tp"] = int(tp)
    cm["fp"] = int(fp)
    cm["tn"] = int(tn)
    cm["fn"] = int(fn)
    cm["kappa"] = float(kappa)
    cm["accuracy"] = float(accuracy)

    return cm


def load_npy_file(file_path):
    try:
        return np.load(file_path)
    except OSError or ValueError or EOFError as err:
        print(err)

