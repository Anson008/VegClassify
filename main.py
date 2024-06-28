import os
import cv2
import numpy as np
import utility
import matplotlib.pyplot as plt
from morphology.filter_factory import FilterFactory
from morphology.connected_components import CV2ConnectedComponentsGenerator, ConnectedComponents
from ndvi.naip_processor import NAIPProcessor
from web_scraper.imagery_wayback_driver import ImageryWaybackDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from Inferencer.deep_recognizer import DeepGreenSpaceRecognizer
from utility.naip_sampler import NaipSampler
from utility.image_block import ImageBlock
from utility import util


# NAIP_RANDOM_SAMPLES_DIR = "./cache/naip_random_samples/"
# NAIP_RANDOM_SAMPLES_MASKS_DIR = "./cache/naip_random_samples_masks/"
# WAYBACK_SCREENSHOTS_DIR = "./cache/wayback_screenshots/"
# GROUND_TRUTH_MASKS_DIR = "./cache/ground_truth_masks/"
# GROUND_TRUTH_IMAGES_DIR = "./cache/ground_truth_images/"


def create_cache():
    util.create_directory(util.WAYBACK_SCREENSHOTS_DIR)
    util.create_directory(util.NAIP_RANDOM_SAMPLES_DIR)
    util.create_directory(util.NAIP_SAMPLE_MASKS_DIR)
    util.create_directory(util.GROUND_TRUTH_MASKS_DIR)
    util.create_directory(util.GROUND_TRUTH_IMAGES_DIR)


def clean_cache():
    util.remove_all_files(util.WAYBACK_SCREENSHOTS_DIR)
    util.remove_all_files(util.NAIP_RANDOM_SAMPLES_DIR)
    util.remove_all_files(util.NAIP_SAMPLE_MASKS_DIR)
    util.remove_all_files(util.GROUND_TRUTH_MASKS_DIR)
    util.remove_all_files(util.GROUND_TRUTH_IMAGES_DIR)


def match_naip_to_ground_truth(naip_mask_path, ground_truth_mask_path):
    naip_file_obj = os.scandir(naip_mask_path)
    ground_truth_file_obj = os.scandir(ground_truth_mask_path)

    for naip, gt in zip(naip_file_obj, ground_truth_file_obj):
        if naip.name.endswith(".png") and gt.name.endswith(".png"):
            naip_mask = cv2.imread(os.path.join(naip_mask_path, naip.name))
            gt_mask = cv2.imread(os.path.join(ground_truth_mask_path, gt.name))


def generate_sample_ground_truth(config_path, checkpoint_path, naip_img_path, wayback_scale=18):
    # Create a NAIP processor
    naip_img = util.read_naip_image(naip_img_path)
    naip = NAIPProcessor(naip_img)
    naip_resolution = abs(naip.get_resolution()[0])
    naip_bgr = naip.get_bgr_naip()

    # Initialize Selenium driver
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # options.binary_location = util.SELENIUM_DRIVER_BINARY_LOC

    # Get a screenshot of the region of interest
    naip_h, naip_w = naip_bgr.shape[:2]
    wayback_resolution = util.WAYBACK_SCALE_TO_RESOLUTION[str(wayback_scale)]

    # naip_sample_xy = utility.get_random_naip_imagery_samples(naip_h, naip_w, n_samples_xy, seed=9527)
    naip_sampler = NaipSampler(naip_h, naip_w)
    naip_sample_xy = naip_sampler.get_grid_samples()
    print(f"Number of blocks: {naip_sampler.get_num_of_samples()}")
    n_digits = len(str(naip_sampler.get_num_of_samples()))
    # print(f"Number of digits: {n_digits}")
    np.save("./cache/naip_random_sample_coordinates.npy", naip_sample_xy)

    # Initialize DeepGreen model
    deep_green = DeepGreenSpaceRecognizer(config_path, checkpoint_path)

    for i, diagonal_xy in enumerate(naip_sample_xy):

        image_block = ImageBlock(diagonal_xy)
        center_abs_x, center_abs_y = image_block.get_absolute_center()
        print(f"Center {i + 1} (x, y): {center_abs_x}, {center_abs_y}")
        tx, ty, bx, by = image_block.get_all_coordinates()
        naip_sample_img = naip.naip_img[:, ty:by+1, tx:bx+1]
        sample_bgr_img = util.naip_to_bgr(naip_sample_img)
        cv2.imwrite(os.path.join(util.NAIP_RANDOM_SAMPLES_DIR, f"sample_{i + 1}.png"), sample_bgr_img)

        # Get center coordinates of current sample
        # center_lon, center_lat = naip.get_lon_lat(row=center_abs_y, col=center_abs_x)
        center_lon, center_lat = naip.get_center_lon_lat((tx, ty), (bx, by))
        print(f"Center Geo {i + 1}: {center_lon}, {center_lat}")

        s_h, s_w = image_block.get_block_size()
        wayback_h, wayback_w = util.get_wayback_shot_size((s_h, s_w),
                                                             naip_resolution,
                                                             wayback_resolution,
                                                             beta=1.1)
        # print(f"Wayback shot size: {wayback_h, wayback_w}")
        driver = webdriver.Chrome(options=options)
        wayback_driver = ImageryWaybackDriver(driver)

        # Load webpage with url
        url = wayback_driver.make_url(center_lon, center_lat, scale=wayback_scale)
        wayback_driver.load_url(url)
        wayback_driver.toggle_off_version_filter()
        wayback_driver.accept_cookies()

        save_wayback = os.path.join(util.WAYBACK_SCREENSHOTS_DIR, f"wayback_shot_{i + 1}.png")
        wayback_img = wayback_driver.take_screenshot(wayback_w, wayback_h,
                                                     save_to_file=save_wayback)
        wayback_driver.close()
        wayback_img = cv2.cvtColor(wayback_img, cv2.COLOR_RGB2BGR)

        # Get inference from the DeepGreen model
        ground_truth_segs = deep_green.infer_batch([wayback_img])

        # Threshold the segmentation results generated by the DeepGreen model
        _, ground_truth_seg = cv2.threshold(ground_truth_segs[0], 0, 255, cv2.THRESH_BINARY)
        # ground_truth_seg_binary = ground_truth_seg.astype(np.bool_)
        # np.save(os.path.join(util.GROUND_TRUTH_MASKS_BINARY_DIR,
        #                      f"ground_truth_mask_binary_{str(i + 1).zfill(n_digits)}.npy"),
        #         ground_truth_seg_binary)

        ground_truth_gray = ground_truth_seg.astype(np.uint8)
        ground_truth_color = NAIPProcessor.set_mask_color(ground_truth_gray, (0, 0, 255))
        combined_ground_truth = cv2.addWeighted(wayback_img, 1, ground_truth_color, 0.5, 0)
        cv2.imwrite(os.path.join(util.GROUND_TRUTH_MASKS_DIR, f"ground_truth_mask_{str(i + 1).zfill(n_digits)}.png"),
                    ground_truth_gray)
        cv2.imwrite(os.path.join(util.GROUND_TRUTH_IMAGES_DIR, f"ground_truth_image_{str(i + 1).zfill(n_digits)}.png"),
                    combined_ground_truth)


def generate_naip_vegetation_masks(naip_img_path: str, coordinate_file_path: str, ndvi_threshold: float):
    naip_img = util.read_naip_image(naip_img_path)
    naip_processor = NAIPProcessor(naip_img)
    sample_coordinates = np.load(coordinate_file_path)
    n_samples = sample_coordinates.shape[0]
    n_digits = len(str(n_samples))

    for i, coordinate in enumerate(sample_coordinates):
        tx, ty, bx, by = coordinate
        mask = naip_processor.generate_vegetation_mask((tx, ty), (bx, by), ndvi_threshold)
        # mask_binary = mask.astype(np.bool_)
        # out_filename = f"naip_mask_binary_{str(i + 1).zfill(n_digits)}.npy"
        # out_path = os.path.join(util.NAIP_MASKS_BINARY_DIR, out_filename)
        # np.save(out_path, mask_binary)

        mask[mask != 0] = 255
        out_filename = f"naip_mask_{str(i + 1).zfill(n_digits)}.png"
        out_path = os.path.join(util.NAIP_SAMPLE_MASKS_DIR, out_filename)
        cv2.imwrite(out_path, mask)


def generate_ground_truth(input_dir, config_path, checkpoint_path):
    try:
        with os.scandir(input_dir) as entries:
            sorted_entries = sorted(entries, key=lambda x: x.name)
    except OSError as err:
        print(err)

    images = []
    for i, entry in enumerate(sorted_entries):
        if entry.is_file():
            images.append(cv2.imread(entry.path))

    deep_green = DeepGreenSpaceRecognizer(config_path, checkpoint_path)
    n_digits = len(str(len(images)))
    for i, image in enumerate(images):
        ground_truth_seg = deep_green.infer_batch([image])[0]
        _, ground_truth_gray = cv2.threshold(ground_truth_seg, 0, 255, cv2.THRESH_BINARY)
        ground_truth_gray = ground_truth_gray.astype(np.uint8)
        ground_truth_color = NAIPProcessor.set_mask_color(ground_truth_gray, (0, 0, 255))
        combined_ground_truth = cv2.addWeighted(image, 1, ground_truth_color, 0.5, 0)
        cv2.imwrite(os.path.join("./cache/ground_truth_masks/", f"ground_truth_mask_{str(i).zfill(n_digits)}.png"), ground_truth_gray)
        cv2.imwrite(os.path.join("./cache/ground_truth_images/", f"ground_truth_image_{str(i).zfill(n_digits)}.png"), combined_ground_truth)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Set model configuration path and checkpoint path
    config_path = "./configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
    checkpoint_path = "./checkpoints/iter_1000.pth"
    naip_img_path = "./image/m_4111118_nw_12_060_20210813.tif"

    # create_cache()
    # clean_cache()
    # generate_sample_ground_truth(config_path, checkpoint_path, naip_img_path, wayback_scale=18)
    #
    # coordinate_file_path = "./cache/naip_random_sample_coordinates.npy"
    # generate_naip_vegetation_masks(naip_img_path, coordinate_file_path, 0.1)

    # naip_img = util.read_naip_image(naip_img_path)
    # naip_processor = NAIPProcessor(naip_img)
    # tm_info = naip_processor.get_template_match_info(util.NAIP_RANDOM_SAMPLES_DIR,
    #                                                  util.WAYBACK_SCREENSHOTS_DIR,
    #                                                  util.WAYBACK_SCALE_TO_RESOLUTION["18"])

    tm_info_path = "./cache/tm_info.npy"
    # util.draw_template_match_region(util.WAYBACK_SCREENSHOTS_DIR, tm_info_path)

    cm = util.get_confusion_matrix(util.NAIP_SAMPLE_MASKS_DIR, util.GROUND_TRUTH_MASKS_DIR, tm_info_path)
    print(f"Confusion matrix: {cm}")

    # main(config_path, checkpoint_path, naip_img_path, n_samples_xy=(8, 10))
    #
    # img_path = "./cache/naip_split"
    # config_path = "./configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
    # checkpoint_path = "./checkpoints/iter_1000.pth"
    # output_dir = "./cache/ground_truth_masks"
    #
    # generate_ground_truth(img_path, config_path, checkpoint_path)
    #
    # stitch_in_dir = "./cache/ground_truth_masks/"
    # stitch_out_dir = "./cache/ground_truth_stitched_mask/"
    # utility.stitch_images(stitch_in_dir, stitch_out_dir)

    # stitch_in_dir = "./cache/ground_truth_images/"
    # stitch_out_dir = "./cache/ground_truth_stitched_image/"
    # utility.stitch_images(stitch_in_dir, stitch_out_dir)

