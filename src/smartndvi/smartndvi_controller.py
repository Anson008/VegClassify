import os
import numpy as np
import pandas as pd
import progressbar
import cv2
import json

from pathlib import Path
from typing import Tuple, Dict, NamedTuple, Any, Optional

from utility.confusion_matrix import ConfusionMatrix
from utility.mask_factory import FullMaskCreator, RandomSampledMask, RandomSampledMaskCreator
from utility.toml import TOML
from utility import util
from naip.naip_imagery import NAIPImagery
from naip.naip_sampler import NaipSampler
from naip.sample_method import GridSample
from Inference.deep_recognizer import DeepGreenSpaceRecognizer
from utility.image_block import ImageBlock
from enum import Enum


class Metrics(Enum):
    kappa = 0
    accuracy = 1


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
                                             "iter_16000.pth")

        # Get cache directories
        cache_root_path = toml_document["General"]["Cache"]["cache_root"]
        ground_truth_mask_root_dir = toml_document["General"]["Cache"]["ground_truth_mask"]
        ground_truth_image_root_dir = toml_document["General"]["Cache"]["ground_truth_image"]
        ground_truth_landcover_root_dir = toml_document["General"]["Cache"]["ground_truth_landcover"]
        naip_sample_mask_dir = toml_document["General"]["Cache"]["naip_sample_mask"]

        # Get output directories
        output_optimal_ndvi_dir = toml_document["General"]["Output"]["optimal_ndvi"]
        land_cover_maps_dir = toml_document["General"]["Output"]["land_cover_maps"]
        vegetation_mask_dir = toml_document["General"]["Output"]["vegetation_mask"]

        sample_coordinate_file_name = f"{os.path.basename(naip_path)}_naip_sample_xy.npy"
        sample_coordinate_file_path = os.path.join(cache_root_path, sample_coordinate_file_name)

        sample_shape = (1024, 512)
        self._generate_grid_naip_sample(naip_path, sample_coordinate_file_path, sample_shape)

        naip_basename = os.path.basename(naip_path)[:-4]
        ground_truth_mask_dir = os.path.join(ground_truth_mask_root_dir, naip_basename)
        ground_truth_image_dir = os.path.join(ground_truth_image_root_dir, naip_basename)
        ground_truth_landcover_dir = os.path.join(ground_truth_landcover_root_dir, naip_basename)
        naip_sample_coordinates = util.load_npy_file(sample_coordinate_file_path)

        self._generate_ground_truth_by_deep_learning(naip_sample_coordinates,
                                                     naip_path,
                                                     model_config_path,
                                                     model_checkpoint_path,
                                                     ground_truth_mask_dir,
                                                     ground_truth_image_dir,
                                                     ground_truth_landcover_dir)

        # Initialize searching parameters
        thresholds = tuple(np.arange(0, 0.41, 0.02))
        max_metrics_value = {k.name: 0 for k in Metrics}
        optimal_ndvi = {k.name: {"metrics": 0, "optimal_ndvi": -1} for k in Metrics}
        n_thresholds = len(thresholds)
        print(f">>> Searching optimal NDVI threshold for {os.path.basename(naip_path)} ...")

        opt_data_array = np.zeros((n_thresholds, len(Metrics) + 1), dtype=np.float64)
        with progressbar.ProgressBar(max_value=n_thresholds) as bar:
            for i in range(n_thresholds):
                util.remove_all_files(naip_sample_mask_dir)
                self._generate_naip_vegetation_masks(naip_path,
                                                     sample_coordinate_file_path,
                                                     naip_sample_mask_dir,
                                                     thresholds[i])

                confusion_matrix = ConfusionMatrix()
                mask_creator = FullMaskCreator(sample_shape[0], sample_shape[1])
                # mask_creator = RandomSampledMaskCreator(height=sample_shape[0],
                #                                        width=sample_shape[1],
                #                                        sample_size=1000,
                #                                        seed=95279527)
                confusion_matrix.compute_on_batch_samples(ground_truth_mask_dir,
                                                          naip_sample_mask_dir,
                                                          mask_creator)
                cm = confusion_matrix.get_confusion_matrix()

                opt_data_array[i, 0] = thresholds[i]
                for metrics in Metrics:
                    opt_data_array[i, metrics.value + 1] = cm[metrics.name]
                    if cm[metrics.name] > max_metrics_value[metrics.name]:
                        # Update maximum metrics value
                        max_metrics_value[metrics.name] = cm[metrics.name]

                        # Record maximum metrics value and corresponding optimal NDVI threshold
                        optimal_ndvi[metrics.name]["metrics"] = cm[metrics.name]
                        optimal_ndvi[metrics.name]["optimal_ndvi"] = thresholds[i]
                bar.update(i)

                for metrics in Metrics:
                    if max_metrics_value[metrics.name] == 1.0:
                        break

        output_optimal_ndvi_path = os.path.join(output_optimal_ndvi_dir,
                                                f"{os.path.basename(naip_path)}_optimal_ndvi.json")
        with open(output_optimal_ndvi_path, "w") as outfile:
            outfile.write(json.dumps(optimal_ndvi, indent=4))

        opt_curve_data_path = os.path.join(output_optimal_ndvi_dir,
                                           f"{os.path.basename(naip_path)}_opt_curve_data.csv")
        df = pd.DataFrame(opt_data_array, columns=["ndvi_threshold", Metrics(0).name, Metrics(1).name])
        df.to_csv(opt_curve_data_path, index=False)

        if land_cover_metrics is not None:
            try:
                metrics_name = Metrics[land_cover_metrics].name
                self._generate_land_cover_maps(naip_path,
                                               optimal_ndvi[metrics_name]["optimal_ndvi"],
                                               land_cover_maps_dir,
                                               vegetation_mask_dir)
            except ValueError as err:
                print(f"{err}: invalid metrics name. Land-cover maps not generated.")


    @staticmethod
    def _generate_grid_naip_sample(naip_path: str,
                                   sample_output_path: str,
                                   sample_shape: Tuple[int, int]) -> None:
        """
        A helper function to generate grid sample coordinates (tx, ty, bx, by) on a NAIP imagery.
        :param naip_path: str, full path of NAIP.
        :param sample_output_path: str, full path of the output file.
        :param sample_shape: tuple of int, specifying (height, width) of the individual sample.
        :return: None.
        """
        if not sample_output_path.endswith(".npy"):
            raise Exception("Output file must be of type .npy")

        naip_img = util.read_naip_image(naip_path)
        naip = NAIPImagery(naip_img)
        naip_h, naip_w = naip.naip_img.shape[1:]
        naip_sampler = NaipSampler(GridSample())
        naip_sample_xy = naip_sampler.get_sample_coordinates((naip_h, naip_w), sample_shape)

        np.save(sample_output_path, naip_sample_xy)

    @staticmethod
    def _generate_ground_truth_by_hsv(naip_sample_xy: np.ndarray,
                                      naip_path: str,
                                      output_mask_dir: str,
                                      output_image_dir: str):
        # Create a NAIP processor
        naip_img = util.read_naip_image(naip_path)
        naip = NAIPImagery(naip_img)

        n_samples = naip_sample_xy.shape[0]
        n_digits = len(str(n_samples))

        print(f">>> Generating ground truth vegetation mask for {os.path.basename(naip_path)} ...")
        with progressbar.ProgressBar(max_value=n_samples) as bar:
            for i in range(n_samples):
                image_block = ImageBlock(naip_sample_xy[i])
                tx, ty, bx, by = image_block.get_all_coordinates()
                naip_sample = naip[:, ty:by+1, tx:bx+1]

                ground_truth_gray = naip_sample.get_vegetation_by_hsv(30, 90)
                _, ground_truth_binary = cv2.threshold(ground_truth_gray, 0, 1, cv2.THRESH_BINARY)
                # ground_truth_binary = ground_truth_segs[0].astype(np.uint8)
                ground_truth_land_cover = naip_sample.generate_vegetation_cover_map(ground_truth_binary)
                out_image_path = os.path.join(output_image_dir, f"ground_truth_image_{str(i + 1).zfill(n_digits)}.png")
                cv2.imwrite(out_image_path, ground_truth_land_cover)

                out_mask_path = os.path.join(output_mask_dir, f"ground_truth_mask_{str(i + 1).zfill(n_digits)}.png")
                cv2.imwrite(out_mask_path, ground_truth_gray)
                bar.update(i)

    @staticmethod
    def _generate_ground_truth_by_deep_learning(naip_sample_xy: np.ndarray,
                                             naip_path: str,
                                             config_path: str,
                                             checkpoint_path: str,
                                             output_mask_dir: str,
                                             output_image_dir: str,
                                             output_landcover_dir: str):
        if not os.path.exists(output_mask_dir):
            util.create_directory(output_mask_dir)
        if not os.path.exists(output_image_dir):
            util.create_directory(output_image_dir)
        if not os.path.exists(output_landcover_dir):
            util.create_directory(output_landcover_dir)

        util.remove_all_files(output_mask_dir)
        util.remove_all_files(output_image_dir)
        util.remove_all_files(output_landcover_dir)

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
                naip_sample = naip[:, ty:by+1, tx:bx+1]
                naip_sample_bgr_img = naip_sample.get_bgr_naip()

                # Get inference from the DeepGreen model
                ground_truth_segs = deep_green.infer_batch([naip_sample_bgr_img])

                # Threshold the segmentation results generated by the DeepGreen model
                _, ground_truth_gray = cv2.threshold(ground_truth_segs[0], 0, 255, cv2.THRESH_BINARY)
                ground_truth_gray = ground_truth_gray.astype(np.uint8)

                ground_truth_binary = ground_truth_segs[0].astype(np.uint8)
                ground_truth_land_cover = naip_sample.generate_vegetation_cover_map(ground_truth_binary)
                out_landcover_path = os.path.join(output_landcover_dir, f"ground_truth_landcover_{str(i + 1).zfill(n_digits)}.png")
                cv2.imwrite(out_landcover_path, ground_truth_land_cover)

                out_mask_path = os.path.join(output_mask_dir, f"ground_truth_mask_{str(i + 1).zfill(n_digits)}.png")
                cv2.imwrite(out_mask_path, ground_truth_gray)

                out_image_path = os.path.join(output_image_dir, f"ground_truth_image_{str(i + 1).zfill(n_digits)}.png")
                cv2.imwrite(out_image_path, naip_sample.get_bgr_naip())

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
        vegetation_mask_integrated = naip.integrate_vegetation_mask(vegetation_mask_binary)

        # vegetation_mask_bgr = naip.set_mask_color(vegetation_mask_binary,
        #                                           pos_colors=(84, 163, 49),
        #                                           neg_colors=(185, 252, 247))
        filename_base = os.path.basename(naip_path)
        vegetation_mask_integrated.rio.to_raster(os.path.join(vegetation_mask_output_dir, f"{filename_base}_vegetation_mask.tif"))

        # cv2.imwrite(os.path.join(vegetation_mask_output_dir, f"{filename_base}_vegetation_mask.tif"),
        #             vegetation_mask_integrated)

        vegetation_cover = naip.generate_vegetation_cover_map(vegetation_mask_binary)
        cv2.imwrite(os.path.join(land_cover_output_dir, f"{filename_base}_land_cover.png"),
                    vegetation_cover)
