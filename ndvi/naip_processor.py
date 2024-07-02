import cv2
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import matplotlib.pyplot as plt
import os
from utility import util
import math
import xarray as xr


class NAIPProcessor:

    def __init__(self, naip_img: xr.DataArray):
        """
        :param naip_img: xarray.DataArray, the input NAIP image
        """
        self._naip_img = naip_img

    @property
    def naip_img(self) -> xr.DataArray:
        """
        :return: xarray.DataArray, raw NAIP image
        """
        return self._naip_img

    @naip_img.setter
    def naip_img(self, new_naip_img: xr.DataArray):
        """
        :param: xarray.DataArray, a new NAIP image
        """
        self._naip_img = new_naip_img

    def get_center(self) -> tuple[int, int]:
        height, width = self.naip_img.shape[1:]
        return (height - 1) // 2, (width - 1) // 2

    def reproject(self, dst_crs: str = "EPSG:4326") -> xr.DataArray:
        """
        Reproject the NAIP imagery into the coordinate system specified by dst_crs.
        :param dst_crs: str, OGC WKT string or Proj.4 string
        :return: xarray.DataArray, reprojected NAIP image
        """
        # Reproject NAIP to dst_crs with the same shape
        _, height, width = self.naip_img.shape
        return self._naip_img.rio.reproject(dst_crs, shape=(height, width))

    def get_lon_lat(self, row: int = 0, col: int = 0) -> tuple[int, int]:
        """
        Get the longitude and latitude of the pixel [row, col] of the NAIP image.
        :param row: int, row index of the target pixel
        :param col: int, column index of the target pixel
        :return:
        """
        naip_ll = self.reproject("EPSG:4326")
        return naip_ll[0, row, col].x.values.item(), naip_ll[0, row, col].y.values.item()

    def get_center_lon_lat(self, top_left: tuple[int, int], bottom_right: tuple[int, int]) -> tuple[int, int]:
        """
        Get the longitude and latitude of the center pixel of a rectangular
        block specified by top_left and bottom_right.
        :param top_left: tuple of int, top left pixel coordinates
        :param bottom_right: tuple of int, bottom right pixel coordinates
        :return: tuple of int, longitude and latitude of the center pixel
        """
        tx, ty = top_left
        bx, by = bottom_right
        naip_slice = NAIPProcessor(self._naip_img[:, ty:by+1, tx:bx+1])
        center_row, center_col = naip_slice.get_center()
        return naip_slice.get_lon_lat(row=center_row, col=center_col)

    def get_resolution(self) -> tuple[int, int]:
        """
        Get the resolution of the NAIP image.
        :return: tuple of int, resolution of the NAIP image along x- and y-axis.
        """
        return self._naip_img.rio.resolution()

    def get_bgr_naip(self) -> np.ndarray:
        """
        Get the BGR image of the NAIP imagery.
        :return: numpy array, 3D array of shape (width, height, color) representing RGB image
        of the original 4-band NAIP image
        """
        # Convert xarray.DataArray to numpy array
        naip_np_arr = self.naip_img.values

        # Extract RGB channels and reorder the color axis from (c, w, h) to (w, h, c)
        naip_rgb = np.moveaxis(naip_np_arr[0:3], 0, -1)

        # Convert RGB to BGR, as BGR is the default color model of OpenCV
        return cv2.cvtColor(naip_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def show_image(window_name: str, img: np.ndarray) -> None:
        """
        :param window_name: str, title of the window displaying the image
        :param img: numpy array, image to display
        :return: None
        """
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    @staticmethod
    def calculate_ndvi(naip_img: xr.DataArray) -> np.ndarray:
        """
        Calculate the normalized difference (NDVI) of the input NAIP image.
        :param naip_img: xarray.DataArray, NAIP image to perform the calculation on
        :return: numpy array, NDVI results
        """
        naip_data = naip_img.values.astype(np.float64)
        return es.normalized_diff(naip_data[3], naip_data[0])

    @staticmethod
    def classify(ndvi: np.ndarray, threshold: float, invert: bool = False):
        """
        :param ndvi: numpy array, NDVI values for an image
        :param threshold: float, the value to distinguish green space and non-green space
        :param invert: bool, set to invert the classification result.
                    Default is False, indicating 1 for green and 0 for non-green.
        :return: numpy array, classified pixel values of the image
        """
        classified_ndvi = np.zeros_like(ndvi, dtype=np.uint8)
        classified_ndvi[ndvi >= threshold] = 1
        classified_ndvi[ndvi < threshold] = 0
        if invert:
            classified_ndvi = np.invert(classified_ndvi.astype(np.bool_)).astype(np.uint8)
        # classified_ndvi[classified_ndvi != 0] = 255
        return classified_ndvi

    @staticmethod
    def set_mask_color(classified_ndvi, colors):
        res = cv2.cvtColor(classified_ndvi, cv2.COLOR_GRAY2BGR)
        # print(f"res shape: {res.shape}")
        res[res[:, :, 0] != 0, 0] = colors[0]
        res[res[:, :, 1] != 0, 1] = colors[1]
        res[res[:, :, 2] != 0, 2] = colors[2]
        return res

    @staticmethod
    def plot_bands(img, cmap, title):
        """
        :param img: numpy array, image to display
        :param cmap: str, name of the color map
        :param title: str, title of the figure
        :return: None
        """

        ep.plot_bands(img,
                      cmap=cmap,
                      scale=False,
                      vmin=-1,
                      vmax=1,
                      title=title)
        plt.show()

    def split_image(self, split_height: int, split_width: int) -> None:
        """
        Split the NAIP image into blocks of size (split_height, split_width). Blocks of smaller size are kept as-is.
        :param split_height: int, height of the blocks
        :param split_width: int, width of the blocks
        :return: None
        """
        util.create_directory(util.NAIP_SPLIT_DIR)
        util.remove_all_files(util.NAIP_SPLIT_DIR)
        input_image = self.get_bgr_naip()
        h, w = input_image.shape[:2]
        splits = [input_image[y:y + split_height, x:x + split_width]
                  for y in range(0, h, split_height)
                  for x in range(0, w, split_width)]
        n_digits = len(str(len(splits)))
        for i, split in enumerate(splits):
            if split is not None:
                filename = f"naip_split_rowSize{math.ceil(w / split_width)}_{str(i).zfill(n_digits)}.png"
                cv2.imwrite(os.path.join(util.NAIP_SPLIT_DIR, filename), split)

    def generate_vegetation_mask(self,
                                 top_left: tuple[int, int],
                                 bottom_right: tuple[int, int],
                                 threshold: float,
                                 invert: bool = False) -> np.ndarray:
        """
        Generate vegetation mask specified by top_left and bottom_right on the NAIP image.
        :param top_left: tuple of int, top left pixel coordinates of the target block
        :param bottom_right: tuple of int, bottom right pixel coordinates of the target block
        :param threshold: float, NDVI threshold to classify pixels
        :param invert: bool, set to invert the classification result.
                    Default is False, indicating 1 for green and 0 for non-green.
        :return: numpy array, vegetation mask representing if the pixels are green or not.
        """
        tx, ty = top_left
        bx, by = bottom_right
        img_block = self._naip_img[:, ty:by+1, tx:bx+1]
        ndvi = self.calculate_ndvi(img_block)
        return self.classify(ndvi, threshold, invert)

    def get_template_match_info(self,
                                naip_samples_path: str,
                                wayback_shot_path: str,
                                wayback_resolution: float) -> np.ndarray:
        """
        Get the template match info between the NAIP samples and the Wayback screenshots (ground truth).
        :param naip_samples_path: str, path to the NAIP samples images
        :param wayback_shot_path: str, path to the Wayback screenshot images
        :param wayback_resolution: float, the resolution of the Wayback screenshots
        :return: numpy array of shape (n_samples, 5). Each row represents a template match result:
                (scale, max_val, max_loc, ndvi_h, ndvi_w).
                Scale - the factor that an original NAIP sample is scaled and leads to the best match result.
                max_val - the metric value for the best template match.
                max_loc - the location where the best template match is found in the Wayback screenshot.
                ndvi_h - height of the NAIP sample that leads to the best match result.
                ndvi_w - width of the NAIP sample that leads to the best match result.
        """
        naip_file_obj = os.scandir(naip_samples_path)
        wayback_shot_file_obj = os.scandir(wayback_shot_path)
        tm_info = []

        for naip, wayback_shot in zip(naip_file_obj, wayback_shot_file_obj):
            if naip.name.endswith(".png") and wayback_shot.name.endswith(".png"):
                naip_img = cv2.imread(os.path.join(naip_samples_path, naip.name))
                wayback_img = cv2.imread(os.path.join(wayback_shot_path, wayback_shot.name))

                scale, max_val, max_loc, ndvi_h, ndvi_w = self._match_template(wayback_img, naip_img, wayback_resolution)
                ox, oy = max_loc
                tm_info.append([scale, ox, oy, ndvi_h, ndvi_w])

        tm_info = np.array(tm_info)
        np.save(os.path.join("./cache/tm_info.npy"), tm_info)

        return tm_info

    def _match_template(self,
                        ground_truth_image: np.ndarray,
                        naip_image: np.ndarray,
                        wayback_resolution: float) -> tuple:
        """
        A helper method to find the best template match for the NAIP samples and Wayback screenshots.
        :param ground_truth_image: numpy array of shape (h, w, c), a Wayback screenshot image (BGR).
        :param naip_image: numpy array of shape (h, w, c), a NAIP sample (BGR).
        :param wayback_resolution: float, the resolution of the Wayback screenshots.
        :return: tuple, info for the best template match.
        """
        tm_scales = util.get_template_matching_scales(abs(self.get_resolution()[0]), wayback_resolution)
        global_max = float('-inf')
        optimal_metrics = tuple()
        for i, scale in enumerate(tm_scales):
            ndvi_best = cv2.resize(naip_image.copy(),
                                   dsize=(0, 0),
                                   fx=scale,
                                   fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
            similarity_res = cv2.matchTemplate(ground_truth_image, ndvi_best, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity_res)
            if max_val > global_max:
                global_max = max_val
                optimal_metrics = (scale, max_val, max_loc, ndvi_best.shape[0], ndvi_best.shape[1])

        return optimal_metrics


if __name__ == "__main__":
    img_path = "../image/m_4111118_nw_12_060_20210813.tif"

    naip_img = util.read_naip_image(img_path)
    naip = NAIPProcessor(naip_img)
    # naip.split_image(1024, 512)
    # shape = naip.naip_img.rio.get_gcps()
    # print(type(shape))
    # print(shape)
    ll = naip.get_resolution()
    print(type(ll))





