import os
import cv2
import numpy as np
from typing import Dict
from naip.naip_imagery import NAIPImagery
from naip.naip_sampler import NaipSampler
from naip.sample_method import GridSample
from utility import util
from utility.image_block import ImageBlock


class TrainDataGenerator:
    def __init__(self, naip_dir: str | None = None):
        """
        Create an instance of TrainDataGenerator.
        :param naip_dir: str, directory of NAIP imagery.
        """
        self._naip_dir = naip_dir

    @property
    def naip_dir(self):
        return self._naip_dir

    @naip_dir.setter
    def naip_dir(self, value: str | None):
        self._naip_dir = value

    def generate_train_data(self, output_dir: str, format: str) -> None:
        """
        Generate training data from NAIP imagery and save the results to output_dir.
        :param output_dir: str, output directory of the training data.
        :return: None.
        """
        if not os.path.exists(self._naip_dir):
            print("NAIP directory does not exist.")
            return

        util.create_directory(output_dir)

        grid_sample = GridSample()
        naip_sampler = NaipSampler(grid_sample)

        with os.scandir(self._naip_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".tif"):
                    naip_image = util.read_naip_image(entry.path)
                    print(f"Processing {entry.path}")
                    naip_obj = NAIPImagery(naip_image)
                    naip_h = naip_obj.naip_img.shape[1]
                    naip_w = naip_obj.naip_img.shape[2]
                    sample_coordinates = naip_sampler.get_sample_coordinates(naip_size=(naip_h, naip_w),
                                                                             sample_shape=(1024, 512))
                    n_samples = sample_coordinates.shape[0]
                    n_digits = len(str(n_samples))
                    filename = entry.name.split(".")[0]

                    sub_dir = os.path.join(output_dir, filename + "_split/")
                    util.create_directory(sub_dir)

                    for i, sample_coordinate in enumerate(sample_coordinates):
                        image_block = ImageBlock(sample_coordinate)
                        tx, ty, bx, by = image_block.get_all_coordinates()
                        naip_sample = naip_obj[:, ty:by+1, tx:bx+1]
                        out_image_path = os.path.join(sub_dir, f"{filename}_s{str(i + 1).zfill(n_digits)}{format}")
                        if format == ".png":
                            naip_sample_bgr = naip_sample.get_bgr_naip()
                            cv2.imwrite(out_image_path, naip_sample_bgr)
                        elif format == ".tif":
                            naip_sample.naip_img.rio.to_raster(out_image_path)


class NormalizationStatistics:
    def __init__(self, train_img_dir: str):
        self._train_img_dir = train_img_dir

    @property
    def img_dir(self):
        return self._train_img_dir

    @img_dir.setter
    def img_dir(self, new_path: str):
        self._train_img_dir = new_path

    def compute_mean_and_std(self, height: int, width: int) -> Dict[str, tuple]:
        x = np.zeros((height, width, 3))
        x_squared = np.zeros((height, width, 3))
        count = 0
        with os.scandir(self._train_img_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".jpg"):
                    pixels = cv2.imread(entry.path).astype(np.float64)
                    x += pixels
                    x_squared += np.square(pixels)
                    count += 1

        mean = np.mean(x / count, axis=(0, 1))
        mean_of_squared = np.mean(x_squared / count, axis=(0, 1))
        std = np.sqrt(mean_of_squared - np.square(mean))
        return {"mean": tuple(mean), 'std': tuple(std)}


if __name__ == "__main__":
    naip_dir = "D:\\NAIP_Raw\\"
    output_dir = "E:\\NAIP_Split_TIF\\"
    train_data_generator = TrainDataGenerator(naip_dir)
    train_data_generator.generate_train_data(output_dir, ".tif")

    # train_img_dir = "D:\\DeepGreenSpace_Train_Data\\Labeled\\Naip_National_Labeled_100_voc\\JPEGImages"
    # norm_statistics = NormalizationStatistics(train_img_dir)
    # res = norm_statistics.compute_mean_and_std(1024, 512)
    # print(res)


