import os

import cv2
import math
import numpy as np
from pathlib import Path

from naip.naip_imagery import NAIPImagery
from utility import util

class ImageEditor:
    @staticmethod
    def split_image(image_path: Path, des_root_dir: Path, split_height: int=1024, split_width: int=512) -> None:
        """
        Split the NAIP image into blocks of size (split_height, split_width). Blocks of smaller size are kept as-is.
        :param image_path: Path, the input image path.
        :param des_root_dir: Path, the output directory of the split images.
        :param split_height: int, height of the blocks.
        :param split_width: int, width of the blocks.
        :return: None
        """
        image_name = image_path.name[:-4]
        output_dir = des_root_dir.joinpath(image_name)
        util.create_directory(output_dir)
        util.remove_all_files(output_dir)

        input_image = cv2.imread(str(image_path), flags=-1)
        h, w = input_image.shape[:2]

        splits = [input_image[y:y + split_height, x:x + split_width]
                  for y in range(0, h, split_height)
                  for x in range(0, w, split_width)]
        n_digits = len(str(len(splits)))
        for i, split in enumerate(splits):
            if split is not None:
                filename = Path(f"{image_name}_split_{str(i).zfill(n_digits)}.png")
                output_path = str(output_dir.joinpath(filename))
                cv2.imwrite(output_path, split)

    @staticmethod
    def load_png(image_path: str):
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint)
        mask[mask != 0] = 255
        return mask

    @staticmethod
    def load_raster(raster_path: str):
        raster = util.read_naip_image(raster_path)
        naip_obj = NAIPImagery(raster)
        naip_img = naip_obj.get_bgr_naip()
        naip_img = cv2.cvtColor(naip_img, cv2.COLOR_BGR2GRAY).astype(np.uint)
        naip_img[naip_img != 0] = 255
        return naip_img

    @staticmethod
    def export_grayscale_raster(raster_dir: Path, output_dir: Path):
        with os.scandir(raster_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".tif"):
                    gray_raster = ImageEditor.load_raster(entry.path)
                    output_path = output_dir.joinpath(entry.name[:-4] + ".png")
                    cv2.imwrite(str(output_path), gray_raster)


if __name__ == "__main__":
    # image_path = Path("D:\\naip_playground2\\output\\land_cover_maps\\m_3510651_se_13_060_20220526.tif_land_cover.png")
    # des_path = Path("D:\\NDVI_Results_Analysis\\Land_cover_maps_split")
    # ImageEditor.split_image(image_path, des_path)

    raster_dir = Path("D:\\Accuracy_Assessment\\vegetation_mask")
    output_dir = Path("D:\\Accuracy_Assessment\\grayscale_vegetation_mask")
    ImageEditor.export_grayscale_raster(raster_dir, output_dir)