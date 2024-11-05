import json
import os
import copy
import cv2
import numpy
import numpy as np
from typing import Dict
from pathlib import Path

from torch.nn.functional import threshold

from naip.naip_imagery import NAIPImagery
from Inference.deep_recognizer import DeepGreenSpaceRecognizer
from naip.naip_sampler import NaipSampler
from utility.image_block import ImageBlock
from utility import util
from naip.sample_method import GridSample
import progressbar
from smartndvi import __app_name__
from smartndvi import cli
import click


# NAIP_RANDOM_SAMPLES_DIR = "./cache/naip_random_samples/"
# NAIP_RANDOM_SAMPLES_MASKS_DIR = "./cache/naip_random_samples_masks/"
# WAYBACK_SCREENSHOTS_DIR = "./cache/wayback_screenshots/"
# GROUND_TRUTH_MASKS_DIR = "./cache/ground_truth_masks/"
# GROUND_TRUTH_IMAGES_DIR = "./cache/ground_truth_images/"

# @click.command("hello")
# @click.version_option("0.1.0", prog_name="hello")
# def hello():
#     click.echo("Hello World")


def main():
    cli.app(prog_name=__app_name__)


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


def generate_naip_vegetation_masks(naip_img_path: str,
                                   coordinate_file_path: str,
                                   naip_output_mask_path: str,
                                   ndvi_threshold: float):
    naip_img = util.read_naip_image(naip_img_path)
    naip = NAIPImagery(naip_img)
    sample_coordinates = np.load(coordinate_file_path)
    n_samples = sample_coordinates.shape[0]
    n_digits = len(str(n_samples))

    for i, coordinate in enumerate(sample_coordinates):
        tx, ty, bx, by = coordinate
        naip_sample = naip[:, ty:by+1, tx:bx+1]
        mask = naip_sample.generate_vegetation_mask(ndvi_threshold)
        mask[mask != 0] = 255
        out_filename = f"naip_mask_{str(i + 1).zfill(n_digits)}.png"
        out_path = os.path.join(naip_output_mask_path, out_filename)
        cv2.imwrite(out_path, mask)


def generate_landcover_maps(naip_path: str,
                            ndvi_threshold: float,
                            output_dir: str):
    util.create_directory(output_dir)
    naip_img = util.read_naip_image(naip_path)
    naip = NAIPImagery(naip_img)
    vegetation_cover = naip.generate_vegetation_cover_map(ndvi_threshold)

    out_filename = f"{os.path.basename(naip_path)}_landcover.png"
    out_path = os.path.join(output_dir, out_filename)
    cv2.imwrite(out_path, vegetation_cover)

def generate_grid_naip_sample(naip_path: str, output_path: str, sample_xy_filename: str):
    if not sample_xy_filename.endswith(".npy"):
        raise Exception("Output file must be of type .npy")

    naip_img = util.read_naip_image(naip_path)
    naip = NAIPImagery(naip_img)
    naip_h, naip_w = naip.naip_img.shape[1:]
    naip_sampler = NaipSampler(GridSample())
    naip_sample_xy = naip_sampler.get_sample_coordinates((naip_h, naip_w), (1024, 512))

    output_path = os.path.join(output_path, sample_xy_filename)
    np.save(output_path, naip_sample_xy)


def generate_ground_truth_from_rgb_naip(naip_sample_xy: numpy.ndarray,
                                        naip_path: str,
                                        config_path: str,
                                        checkpoint_path: str,
                                        output_mask_dir: str,
                                        output_image_dir: str):
    if not util.create_directory(output_mask_dir):
        util.remove_all_files(output_mask_dir)

    if not util.create_directory(output_image_dir):
        util.remove_all_files(output_image_dir)

    # Create a NAIP processor
    naip_img = util.read_naip_image(naip_path)
    naip = NAIPImagery(naip_img)

    # Initialize DeepGreen model
    deep_green = DeepGreenSpaceRecognizer(config_path, checkpoint_path)
    n_samples = naip_sample_xy.shape[0]
    n_digits = len(str(n_samples))

    print(f">>> Generating ground truth land cover maps for {os.path.basename(naip_path)} ...")
    with progressbar.ProgressBar(max_value=n_samples) as bar:
        for i in range(n_samples):
            image_block = ImageBlock(naip_sample_xy[i])
            tx, ty, bx, by = image_block.get_all_coordinates()
            naip_sample = naip[:, ty:by + 1, tx:bx + 1]
            naip_sample_bgr_img = naip_sample.get_bgr_naip()

            # Get inference from the DeepGreen model
            ground_truth_segs = deep_green.infer_batch([naip_sample_bgr_img])

            # Threshold the segmentation results generated by the DeepGreen model
            _, ground_truth_gray = cv2.threshold(ground_truth_segs[0], 0, 255, cv2.THRESH_BINARY)
            ground_truth_gray = ground_truth_gray.astype(np.uint8)

            ground_truth_binary = ground_truth_segs[0].astype(np.uint8)
            ground_truth_color = NAIPImagery.set_mask_color(ground_truth_binary, (0, 0, 255))
            combined_ground_truth = cv2.addWeighted(naip_sample_bgr_img, 1, ground_truth_color, 0.5, 0)

            out_mask_path = os.path.join(output_mask_dir, f"ground_truth_mask_{str(i + 1).zfill(n_digits)}.png")
            out_image_path = os.path.join(output_image_dir, f"ground_truth_image_{str(i + 1).zfill(n_digits)}.png")
            cv2.imwrite(out_mask_path, ground_truth_gray)
            cv2.imwrite(out_image_path, combined_ground_truth)
            bar.update(i)

def init_workspace(work_dir_root: str):
    workspace = {"cache_dir": os.path.join(work_dir_root, "cache"),
                 "landcover_dir": os.path.join(work_dir_root, "landcover_maps"),
                 "optimal_ndvi_dir": os.path.join(work_dir_root, "optimal_ndvi"),
                 }

    workspace["ground_truth_mask_dir"] = os.path.join(workspace["cache_dir"], "ground_truth_mask")
    workspace["ground_truth_image_dir"] = os.path.join(workspace["cache_dir"], "ground_truth_image")
    workspace["naip_sample_mask_dir"] = os.path.join(workspace["cache_dir"], "naip_sample_mask")

    util.create_directory(work_dir_root)
    util.create_directory(workspace["cache_dir"])
    util.create_directory(workspace["landcover_dir"])
    util.create_directory(workspace["optimal_ndvi_dir"])
    util.create_directory(workspace["ground_truth_mask_dir"])
    util.create_directory(workspace["ground_truth_image_dir"])
    util.create_directory(workspace["naip_sample_mask_dir"])

    return workspace

def find_optimal_ndvi_by_naip(naip_path: str, workspace: Dict[str, str]) -> None:

    # Set model configuration path and checkpoint path
    config_path = "..\\configs\\fcn_aux-hr48_256x512_80k_singlegreen.py"
    checkpoint_path = "..\\..\\models\\iter_1000.pth"
    # naip_path = "..\\..\\image\\m_4111118_nw_12_060_20210813.tif"
    sample_coordinate_file_name = f"{os.path.basename(naip_path)}_naip_sample_xy.npy"

    generate_grid_naip_sample(naip_path, workspace["cache_dir"], sample_coordinate_file_name)

    sample_coordinate_file_path = os.path.join(workspace["cache_dir"], sample_coordinate_file_name)
    naip_sample_coordinates = util.load_npy_file(sample_coordinate_file_path)
    generate_ground_truth_from_rgb_naip(naip_sample_coordinates,
                                        naip_path,
                                        config_path,
                                        checkpoint_path,
                                        workspace["ground_truth_mask_dir"],
                                        workspace["ground_truth_image_dir"])

    thresholds = tuple(np.arange(0, 0.4, 0.02))

    metrics_names = ("accuracy", "kappa")
    max_metrics_value = {k:0 for k in metrics_names}
    optimal_ndvi = {k:{"metrics": 0, "optimal_ndvi": -1} for k in metrics_names}

    # naip_path = Path(naip_path)
    cm_outfile_name = f"{os.path.basename(naip_path)}_optimal_ndvi.json"
    cm_outfile_path = os.path.join(workspace["optimal_ndvi_dir"], cm_outfile_name)

    # util.create_directory(naip_sample_mask_dir)

    n_thresholds = len(thresholds)
    print(f">>> Searching optimal NDVI threshold for {os.path.basename(naip_path)} ...")
    with progressbar.ProgressBar(max_value=n_thresholds) as bar:
        for i in range(n_thresholds):
            util.remove_all_files(workspace["naip_sample_mask_dir"])
            generate_naip_vegetation_masks(naip_path,
                                           sample_coordinate_file_path,
                                           workspace["naip_sample_mask_dir"],
                                           thresholds[i])

            cm = util.get_confusion_matrix_on_naip(workspace["naip_sample_mask_dir"],
                                                   workspace["ground_truth_mask_dir"])

            for metrics_name in metrics_names:
                if cm[metrics_name] > max_metrics_value[metrics_name]:
                    # Update maximum metrics value
                    max_metrics_value[metrics_name] = cm[metrics_name]

                    # Record maximum metrics value and corresponding optimal NDVI threshold
                    optimal_ndvi[metrics_name]["metrics"] = cm[metrics_name]
                    optimal_ndvi[metrics_name]["optimal_ndvi"] = thresholds[i]
            bar.update(i)
    with open(cm_outfile_path, "w") as outfile:
        outfile.write(json.dumps(optimal_ndvi, indent=4))

    generate_landcover_maps(naip_path,
                            optimal_ndvi["kappa"]["optimal_ndvi"],
                            workspace["landcover_dir"])

def batch_process_naip(naip_dir: str, workspace: Dict[str, str]) -> None:
    print(f">>> Start batch processing at {naip_dir} <<<")
    with os.scandir(naip_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".tif"):
                find_optimal_ndvi_by_naip(entry.path, workspace)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # Set model configuration path and checkpoint path
    # config_path = "../configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
    # checkpoint_path = "../../models/iter_1000.pth"
    # naip_path = "../../image/m_4111118_nw_12_060_20210813.tif"
    #
    # generate_grid_naip_sample(naip_path, "../../cache/", "naip_sample_xy.npy")
    #
    # sample_coordinate_file_path = "../../cache/naip_sample_xy.npy"
    # naip_sample_coordinates = util.load_npy_file(sample_coordinate_file_path)
    # generate_ground_truth_from_rgb_naip(naip_sample_coordinates,
    #                                     naip_path,
    #                                     config_path,
    #                                     checkpoint_path,
    #                                     "./cache/ground_truth_mask2/",
    #                                     "./cache/ground_truth_image2/")

    # workspace = init_workspace("D:\\naip_results_batch")

    # Process a single NAIP imagery
    # naip_path = "../../image/m_4111118_nw_12_060_20210813.tif"
    # find_optimal_ndvi_by_naip(naip_path, workspace)

    # Process a batch of NAIP imagery
    # naip_dir = "D:\\naip_test"
    # batch_process_naip(naip_dir)

    main()


