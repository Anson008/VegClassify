import cv2
import numpy as np

from ndvi.naip_processor import NAIPProcessor
from web_scraper.imagery_wayback_driver import ImageryWaybackDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from mmseg.apis import init_model, inference_model, show_result_pyplot
import matplotlib.pyplot as plt


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


def generate_train_data():
    # top_left = (41.763356, -111.860899)
    # bottom_right = (41.701922, -111.801124)
    # xy_min = [min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])]
    # xy_max = [max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])]
    n_samples = 1
    location_data = np.zeros((n_samples, 2))
    location_data[0, 0] = 40.68889
    location_data[0, 1] = -111.86859
    # print(location_data)

    # Specify Chrome driver path
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    # Start Chrome
    # driver = webdriver.Chrome(options=options)

    base_url = "https://livingatlas.arcgis.com/wayback/#active={:d}&mapCenter={:.6f}%2C{:.6f}%2C{:d}"
    release_num = 47963
    scale_factor = 18
    width = 512
    height = 1024

    out_path_base = f"./image/test_data/RN{release_num}_C{scale_factor}_{width}X{height}_Num{n_samples}"
    if not os.path.exists(out_path_base):
        os.makedirs(out_path_base)

    base_filename = "green_space_"
    suffix = ".png"

    pred_out_path = "../image/predictions"
    for i in range(location_data.shape[0]):
        driver = webdriver.Chrome(options=options)
        wayback = ImageryWaybackDriver(driver)
        url = base_url.format(release_num, location_data[i, 1], location_data[i, 0], scale_factor)
        wayback.load_url(url)
        wayback.toggle_off_version_filter()
        wayback.accept_cookies()
        filename = base_filename + str(i).zfill(len(str(n_samples))) + suffix
        save_to_file = os.path.join(out_path_base, filename)
        wayback.take_screenshot(width, height, save_to_file)

        config_path = "../configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
        checkpoint_path = "../checkpoints/iter_1000.pth"

        test_img = save_to_file
        pred_out_filename = "pred_{}.png".format(filename)
        pred_img = os.path.join(pred_out_path, pred_out_filename)

        model = init_model(config_path, checkpoint_path, device="cuda:0")
        pred_res = inference_model(model, test_img)

        vis_image = show_result_pyplot(model, test_img, pred_res, out_file=pred_img, wait_time=1)

        print("Done!")


if __name__ == "__main__":
    config_path = "../configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
    checkpoint_path = "../checkpoints/iter_1000.pth"

    naip_img_path = "../image/m_4111118_nw_12_060_20210813_Clip.tif"
    naip = NAIPProcessor(naip_img_path)
    naip_bgr = naip.get_bgr_naip()

    center_r, center_c = naip.get_center()
    center_lon, center_lat = naip.get_lon_lat(row=center_r, col=center_c)

    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    driver = webdriver.Chrome(options=options)
    wayback_driver = ImageryWaybackDriver(driver)
    url = wayback_driver.make_url(center_lon, center_lat, scale=17)
    wayback_driver.load_url(url)
    wayback_driver.toggle_off_version_filter()
    wayback_driver.accept_cookies()
    # save_wayback_img = "../image/test_data/test1/wayback_img.jpg"
    wayback_img = wayback_driver.take_screenshot(512, 1024)

    # test_img_path = "../image/"
    # test_img_names = ["m_4111118_nw_12_060_20210813_Clip.tif"]
    out_path_base = "../image/test_data/test1/output_wayback/"
    #
    #
    deep_green = Inferencer(config_path, checkpoint_path)
    ground_truth_segs = deep_green.infer_batch([wayback_img])

    ground_truth_seg = ground_truth_segs[0].astype(np.uint8)
    ground_truth_seg[ground_truth_seg != 0] = 255
    print(f"Ground truth: {ground_truth_seg.max(), ground_truth_seg.min()}")

    naip_reprojected = naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)
    template_h, template_w = ndvi.shape[:2]

    start = 0
    end = 0.3
    step = 0.02
    thresholds = np.arange(start, end + step, step)
    best_threshold = dict()
    optimal_metrics = dict()

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for method in methods:
        global_min = float('inf')
        global_max = float('-inf')
        print(f"Matching on {method}")
        for threshold in thresholds:
            ndvi_classified = NAIPProcessor.classify(ndvi, threshold, invert=False)

            similarity_res = cv2.matchTemplate(ground_truth_seg, ndvi_classified, eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity_res)

            if "SQDIFF" in method:
                if min_val < global_min:
                    global_min = min_val
                    best_threshold[method] = (threshold, min_val, min_loc, similarity_res)
            else:
                if max_val > global_max:
                    global_max = max_val
                    best_threshold[method] = (threshold, max_val, max_loc, similarity_res)

    if not os.path.exists(out_path_base):
        try:
            os.mkdir(out_path_base)
        except OSError as error:
            print(error)

    color_ground_truth = NAIPProcessor.set_mask_color(ground_truth_seg, (0, 0, 255))
    combined_ground_truth = cv2.addWeighted(wayback_img, 1, color_ground_truth, 0.5, 0)
    cv2.imwrite(os.path.join(out_path_base, "ground_truth_wayback.png"), combined_ground_truth)

    for key, val in best_threshold.items():
        print(f"{key}: {val[:2]}")
        ground_truth_copy = combined_ground_truth.copy()
        best_ndvi_img = NAIPProcessor.classify(ndvi, val[0], invert=False)
        color_mask = NAIPProcessor.set_mask_color(best_ndvi_img, (0, 0, 255))
        bottom_right = (val[2][0] + template_w, val[2][1] + template_h)
        cv2.rectangle(ground_truth_copy, val[2], bottom_right, (0, 255, 255), 2)
        filename = "Detected_point_{}.png".format(key)
        cv2.imwrite(os.path.join(out_path_base, filename), ground_truth_copy)

        alpha = 0.5
        combined_img = cv2.addWeighted(naip_bgr, 1, color_mask, 0.5, 0)
        filename = "recogition_wayback_{}.png".format(key)
        full_path = os.path.join(out_path_base, filename)
        cv2.imwrite(full_path, combined_img)

    # print(naip_bgr.shape)
