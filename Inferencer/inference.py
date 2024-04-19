import cv2
import numpy as np
import util
from ndvi.naip_processor import NAIPProcessor
from web_scraper.imagery_wayback_driver import ImageryWaybackDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from mmseg.apis import init_model, inference_model, show_result_pyplot
import matplotlib.pyplot as plt


wayback_scale_to_resolution = {"16": 0.9843, "18": 0.2237, "17": 0.4474}


class Inferencer:

    PRED_SEG_STR = "pred_sem_seg"

    def __init__(self, config_path, checkpoint_path):
        # config_path = "../configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
        # checkpoint_path = "../checkpoints/iter_1000.pth"

        self.model = init_model(config_path, checkpoint_path, device="cuda:0")
        self.test_image_directory = None
        self.test_image_names = None

    def get_image_list(self, test_image_directory, test_image_names):
        self.test_image_directory = test_image_directory
        self.test_image_names = test_image_names
        image_list = []
        for test_image_name in test_image_names:
            image_list.append(cv2.imread(os.path.join(test_image_directory, test_image_name)))
            # base_name = os.path.splitext(test_image_name)[0]
            # full_out_paths.append(os.path.join(output_directory, base_name))

        return image_list

    def infer_batch(self, images):
        # Inference on a list of images
        inference = inference_model(self.model, images)

        # Extract numpy array of the predicted segmentation map
        seg_maps = []
        for result in inference:
            for item in result.numpy().items():
                if item[0] == Inferencer.PRED_SEG_STR:
                    seg_gray = np.squeeze(item[1].data, axis=0).astype(np.float32)
                    seg_maps.append(seg_gray)
                    # self.show_seg_map(seg_gray)
        return seg_maps

    # def infer(self, image):
    #     inference = inference_model(self.model, image)
    #
    #     # Extract numpy array of the predicted segmentation map
    #     seg_map = None
    #     for item in inference.numpy().items():
    #         if item[0] == Inferencer.PRED_SEG_STR:
    #             seg_map = np.squeeze(item[1].data, axis=0).astype(np.float32)
    #
    #     return seg_map

    @staticmethod
    def show_seg_map(seg_map):
        ret, thresh1 = cv2.threshold(seg_map, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow('Segmentation Map', thresh1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_wayback_shot_size(naip_size, naip_resolution, wayback_resolution, beta):
    naip_h, naip_w = naip_size
    wayback_w = round(naip_h * naip_resolution / wayback_resolution * beta)
    wayback_h = round(naip_w * naip_resolution / wayback_resolution * beta)
    if wayback_w <= naip_w:
        wayback_w = round(naip_w * beta)
    if wayback_h <= naip_h:
        wayback_h = round(naip_h * beta)
    return wayback_h, wayback_w


if __name__ == "__main__":
    # Set model configuration path and checkpoint path
    config_path = "../configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
    checkpoint_path = "../checkpoints/iter_1000.pth"

    # Set input NAIP imagery path
    naip_img_path = "../image/m_4111118_nw_12_060_20210813_Clip.tif"

    # Create a NAIP processor
    naip = NAIPProcessor(naip_img_path)
    naip_bgr = naip.get_bgr_naip()

    center_r, center_c = naip.get_center()
    center_lon, center_lat = naip.get_lon_lat(row=center_r, col=center_c)

    # Initialize Selenium driver
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    driver = webdriver.Chrome(options=options)
    wayback_driver = ImageryWaybackDriver(driver)

    # Load webpage with url
    scale = 18
    url = wayback_driver.make_url(center_lon, center_lat, scale=scale)
    wayback_driver.load_url(url)
    wayback_driver.toggle_off_version_filter()
    wayback_driver.accept_cookies()
    # save_wayback_img = "../image/test_data/test1/wayback_img.jpg"

    # Get a screenshot of the region of interest
    naip_h, naip_w = naip_bgr.shape[:2]
    wayback_resolution = wayback_scale_to_resolution[str(scale)]
    naip_resolution = abs(naip.get_resolution()[0])

    beta = 1.1  # Screenshot scale factor
    wayback_h, wayback_w = get_wayback_shot_size((naip_h, naip_w), naip_resolution, wayback_resolution, beta)

    wayback_img = wayback_driver.take_screenshot(wayback_w, wayback_h)
    wayback_img = cv2.cvtColor(wayback_img, cv2.COLOR_RGB2BGR)

    # Set output root directory
    out_path_base = f"../image/test_data/test_morph_open_close/output_wayback{wayback_h}X{wayback_w}_GTScale{scale}_TMScaleDYN/"

    # Get inference from the DeepGreen model
    deep_green = Inferencer(config_path, checkpoint_path)
    ground_truth_segs = deep_green.infer_batch([wayback_img])

    # Threshold the segmentation results generated by the DeepGreen model
    ground_truth_seg = ground_truth_segs[0].astype(np.uint8)
    ground_truth_seg[ground_truth_seg != 0] = 255
    # print(f"Ground truth size: {ground_truth_seg.shape}")

    # Calculate NDVI on the NAIP imagery
    naip_reprojected = naip.naip_img  #naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)

    # Alpha is the resizing factor that guarantees a Wayback imagery screenshot
    # having the same resolution as NAIP imagery
    alpha = naip_resolution / wayback_resolution
    # print(f"alpha={alpha}")
    ndvi_thresholds = np.linspace(0, 0.3, 30, endpoint=True)
    delta = alpha * (beta - 1)
    # print(f"wayback_h * beta = {wayback_h * beta}")
    # print(f"naip_h * alpha = {naip_h * alpha}")
    # print(f"delta={delta}")

    # Num must be an even number
    scales = np.linspace(alpha - delta, alpha + delta, 4, endpoint=False)
    best_threshold = dict()
    optimal_metrics = dict()

    # Set template matching methods
    # methods_backup = ['cv2.TM_CCOEFF', 'cv2.TM_CCORR', 'cv2.TM_SQDIFF']
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']
    for method in methods:
        global_min = float('inf')
        global_max = float('-inf')
        print(f"Matching on {method}")
        for threshold in ndvi_thresholds:
            ndvi_classified = NAIPProcessor.classify(ndvi, threshold, invert=False)
            ndvi_classified = cv2.morphologyEx(ndvi_classified, cv2.MORPH_OPEN, open_kernel)
            ndvi_classified = cv2.morphologyEx(ndvi_classified, cv2.MORPH_CLOSE, close_kernel)
            for scale in scales:
                ndvi_temp = ndvi_classified.copy()
                ndvi_temp = cv2.resize(ndvi_temp,
                                             dsize=(0, 0),
                                             fx=scale,
                                             fy=scale,
                                             interpolation=cv2.INTER_CUBIC)
                # template_h, template_w = ndvi_classified.shape[:2]
                similarity_res = cv2.matchTemplate(ground_truth_seg, ndvi_temp, eval(method))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity_res)

                if "SQDIFF" in method:
                    if min_val < global_min:
                        global_min = min_val
                        best_threshold[method] = (threshold, min_val, min_loc, scale, ndvi_temp.shape[:2], similarity_res)
                else:
                    if max_val > global_max:
                        global_max = max_val
                        best_threshold[method] = (threshold, max_val, max_loc, scale, ndvi_temp.shape[:2], similarity_res)

    if not os.path.exists(out_path_base):
        try:
            os.makedirs(out_path_base)
            print("Directory created.")
        except OSError as error:
            print(error)

    wayback_img_filename = f"wayback_screenshot_{wayback_h}X{wayback_w}.png"
    cv2.imwrite(os.path.join(out_path_base, wayback_img_filename), wayback_img)

    color_ground_truth = NAIPProcessor.set_mask_color(ground_truth_seg, (0, 0, 255))
    combined_ground_truth = cv2.addWeighted(wayback_img, 1, color_ground_truth, 0.5, 0)
    cv2.imwrite(os.path.join(out_path_base, "ground_truth_wayback.png"), combined_ground_truth)

    for key, val in best_threshold.items():
        print(f"{key}: {val[:-1]}")
        ground_truth_copy = combined_ground_truth.copy()
        best_ndvi_img = NAIPProcessor.classify(ndvi, val[0], invert=False)
        best_ndvi_img = cv2.morphologyEx(best_ndvi_img, cv2.MORPH_OPEN, open_kernel)
        best_ndvi_img = cv2.morphologyEx(best_ndvi_img, cv2.MORPH_CLOSE, close_kernel)
        color_mask = NAIPProcessor.set_mask_color(best_ndvi_img, (0, 0, 255))
        bottom_right = (val[2][0] + val[4][1], val[2][1] + val[4][0])
        cv2.rectangle(ground_truth_copy, val[2], bottom_right, (0, 255, 255), 2)
        filename = "Detected_point_{}.png".format(key)
        cv2.imwrite(os.path.join(out_path_base, filename), ground_truth_copy)

        alpha = 0.5
        combined_img = cv2.addWeighted(naip_bgr, 1, color_mask, 0.5, 0)
        filename = "recogition_wayback_{}.png".format(key)
        full_path = os.path.join(out_path_base, filename)
        cv2.imwrite(full_path, combined_img)

    # print(naip_bgr.shape)
