import os
import numpy as np
import progressbar
import cv2
import json
from smartndvi import DB_READ_ERROR, FILE_ERROR
from pathlib import Path
from typing import List, Dict, NamedTuple, Any
from utility.toml import TOML
from utility import util
from naip.naip_imagery import NAIPImagery
from naip.naip_sampler import NaipSampler
from naip.sample_method import GridSample
from Inference.deep_recognizer import DeepGreenSpaceRecognizer
from utility.image_block import ImageBlock
from enum import Enum


class Metrics(Enum):
    KAPPA = "kappa"
    ACCURACY = "accuracy"


class CurrentOptimalNDVI(NamedTuple):
    results: Dict[str, Any]
    error: int


class SmartNDVIController:
    def __init__(self, config_path: Path) -> None:
        self._toml = TOML(config_path)

    def optimize_ndvi_threshold(self, naip_path: str, land_cover_metrics: str):
        if not os.path.exists(naip_path):
            raise OSError
        elif os.path.isfile(naip_path):
            print(f">>> Start processing '{os.path.basename(naip_path)}'")
            self._find_optimal_ndvi_threshold(naip_path, land_cover_metrics)
        elif os.path.isdir(naip_path):
            print(f">>> Start batch processing in '{naip_path}")
            with os.scandir(naip_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".tif"):
                        self._find_optimal_ndvi_threshold(entry.path, land_cover_metrics)


    def _find_optimal_ndvi_threshold(self, naip_path: str, land_cover_metrics: str):
        toml_document = self._toml.toml_document

        # Get model config and checkpoint
        model_config_path = os.path.join(toml_document["General"]["Model"]["config"],
                                         "fcn_aux-hr48_256x512_80k_singlegreen.py")
        model_checkpoint_path = os.path.join(toml_document["General"]["Model"]["checkpoint"],
                                             "iter_1000.pth")

        # Get cache directories
        cache_root_path = toml_document["General"]["Cache"]["cache_root"]
        ground_truth_mask_dir = toml_document["General"]["Cache"]["ground_truth_mask"]
        ground_truth_image_dir = toml_document["General"]["Cache"]["ground_truth_image"]
        naip_sample_mask_dir = toml_document["General"]["Cache"]["naip_sample_mask"]

        # Get output directories
        output_optimal_ndvi_dir = toml_document["General"]["Output"]["optimal_ndvi"]
        land_cover_maps_dir = toml_document["General"]["Output"]["land_cover_maps"]
        vegetation_mask_dir = toml_document["General"]["Output"]["vegetation_mask"]

        sample_coordinate_file_name = f"{os.path.basename(naip_path)}_naip_sample_xy.npy"
        sample_coordinate_file_path = os.path.join(cache_root_path, sample_coordinate_file_name)

        self._generate_grid_naip_sample(naip_path, sample_coordinate_file_path)

        naip_sample_coordinates = util.load_npy_file(sample_coordinate_file_path)
        self._generate_ground_truth_from_rgb_naip(naip_sample_coordinates,
                                                  naip_path,
                                                  model_config_path,
                                                  model_checkpoint_path,
                                                  ground_truth_mask_dir,
                                                  ground_truth_image_dir)

        # Initialize searching parameters
        thresholds = tuple(np.arange(0, 0.41, 0.02))
        max_metrics_value = {k.value: 0 for k in Metrics}
        optimal_ndvi = {k.value: {"metrics": 0, "optimal_ndvi": -1} for k in Metrics}
        n_thresholds = len(thresholds)
        print(f">>> Searching optimal NDVI threshold for {os.path.basename(naip_path)} ...")

        with progressbar.ProgressBar(max_value=n_thresholds) as bar:
            for i in range(n_thresholds):
                util.remove_all_files(naip_sample_mask_dir)
                self._generate_naip_vegetation_masks(naip_path,
                                                     sample_coordinate_file_path,
                                                     naip_sample_mask_dir,
                                                     thresholds[i])

                cm = util.get_confusion_matrix_on_naip(naip_sample_mask_dir,
                                                       ground_truth_mask_dir)

                for metrics in Metrics:
                    metrics_name = metrics.value
                    if cm[metrics_name] > max_metrics_value[metrics_name]:
                        # Update maximum metrics value
                        max_metrics_value[metrics_name] = cm[metrics_name]

                        # Record maximum metrics value and corresponding optimal NDVI threshold
                        optimal_ndvi[metrics_name]["metrics"] = cm[metrics_name]
                        optimal_ndvi[metrics_name]["optimal_ndvi"] = thresholds[i]
                bar.update(i)
        output_optimal_ndvi_path = os.path.join(output_optimal_ndvi_dir,
                                                f"{os.path.basename(naip_path)}_optimal_ndvi.json")
        with open(output_optimal_ndvi_path, "w") as outfile:
            outfile.write(json.dumps(optimal_ndvi, indent=4))

        if land_cover_metrics is not None:
            try:
                metrics_name = Metrics(land_cover_metrics).value
                self._generate_land_cover_maps(naip_path,
                                               optimal_ndvi[metrics_name]["optimal_ndvi"],
                                               land_cover_maps_dir,
                                               vegetation_mask_dir)
            except ValueError as err:
                print(f"{err}: invalid metrics name. Land-cover maps not generated.")


    @staticmethod
    def _generate_grid_naip_sample(naip_path: str, sample_output_path: str):
        if not sample_output_path.endswith(".npy"):
            raise Exception("Output file must be of type .npy")

        naip_img = util.read_naip_image(naip_path)
        naip = NAIPImagery(naip_img)
        naip_h, naip_w = naip.naip_img.shape[1:]
        naip_sampler = NaipSampler(GridSample())
        naip_sample_xy = naip_sampler.get_sample_coordinates((naip_h, naip_w), (1024, 512))

        np.save(sample_output_path, naip_sample_xy)

    @staticmethod
    def _generate_ground_truth_from_rgb_naip(naip_sample_xy: np.ndarray,
                                             naip_path: str,
                                             config_path: str,
                                             checkpoint_path: str,
                                             output_mask_dir: str,
                                             output_image_dir: str):
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

                # ground_truth_binary = ground_truth_segs[0].astype(np.uint8)
                # ground_truth_land_cover = naip_sample.generate_vegetation_cover_map(ground_truth_binary)
                # out_image_path = os.path.join(output_image_dir, f"ground_truth_image_{str(i + 1).zfill(n_digits)}.png")
                # cv2.imwrite(out_image_path, ground_truth_land_cover)

                out_mask_path = os.path.join(output_mask_dir, f"ground_truth_mask_{str(i + 1).zfill(n_digits)}.png")
                cv2.imwrite(out_mask_path, ground_truth_gray)
                bar.update(i)

    @staticmethod
    def _generate_naip_vegetation_masks(naip_img_path: str,
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
            naip_sample = naip[:, ty:by + 1, tx:bx + 1]
            mask = naip_sample.generate_vegetation_mask(ndvi_threshold)
            mask[mask != 0] = 255
            out_filename = f"naip_mask_{str(i + 1).zfill(n_digits)}.png"
            out_path = os.path.join(naip_output_mask_path, out_filename)
            cv2.imwrite(out_path, mask)

    @staticmethod
    def _generate_land_cover_maps(naip_path: str,
                                  ndvi_threshold: float,
                                  land_cover_output_dir: str,
                                  vegetation_mask_output_dir: str):
        naip_img = util.read_naip_image(naip_path)
        naip = NAIPImagery(naip_img)
        vegetation_mask_binary = naip.generate_vegetation_mask(ndvi_threshold)

        vegetation_mask_bgr = naip.set_mask_color(vegetation_mask_binary,
                                                  pos_colors=(84, 163, 49),
                                                  neg_colors=(185, 252, 247))
        vegetation_cover = naip.generate_vegetation_cover_map(vegetation_mask_binary)

        filename_base = os.path.basename(naip_path)
        cv2.imwrite(os.path.join(vegetation_mask_output_dir, f"{filename_base}_vegetation_mask.png"),
                    vegetation_mask_bgr)
        cv2.imwrite(os.path.join(land_cover_output_dir, f"{filename_base}_land_cover.png"),
                    vegetation_cover)
