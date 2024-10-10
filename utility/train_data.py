from naip.naip_processor import NAIPImagery
from naip.naip_sampler import NaipSampler
from naip.sample_method import GridSample
from utility import util
import os
import cv2
from utility.image_block import ImageBlock


class TrainDataGenerator:
    def __init__(self, naip_dir: str | None = None):
        self._naip_dir = naip_dir

    @property
    def naip_dir(self):
        return self._naip_dir

    @naip_dir.setter
    def naip_dir(self, value: str | None):
        self._naip_dir = value

    def generate_train_data(self, output_dir: str):
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
                        # print(f"Processing sample {i + 1}")
                        image_block = ImageBlock(sample_coordinate)
                        tx, ty, bx, by = image_block.get_all_coordinates()
                        naip_sample = naip_obj[:, ty:by, tx:bx]
                        naip_sample_bgr = naip_sample.get_bgr_naip()
                        out_image_path = os.path.join(sub_dir, f"{filename}_s{str(i + 1).zfill(n_digits)}.png")
                        cv2.imwrite(out_image_path, naip_sample_bgr)


if __name__ == "__main__":
    naip_dir = "D:/NAIP/"
    output_dir = "D:/naip_split/"
    train_data_generator = TrainDataGenerator(naip_dir)
    train_data_generator.generate_train_data(output_dir)


