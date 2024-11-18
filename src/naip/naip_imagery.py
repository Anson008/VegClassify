from typing import Tuple
import cv2
import earthpy.spatial as es
import numpy as np
import os
from utility import util
import math
import xarray as xr


class NAIPImagery:

    def __init__(self, naip_img: xr.DataArray):
        """
        :param naip_img: xarray.DataArray, the input NAIP image
        """
        self._naip_img = naip_img
        self._ndvi = None

    def __getitem__(self, index):
        return NAIPImagery(self._naip_img[index])

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

    @property
    def ndvi(self) -> np.ndarray | None:
        return self._ndvi

    def get_center(self) -> tuple[int, int]:
        """
        Get the center coordinates of the NAIP imagery.
        :return: tuple of int, (center_y, center_x).
        """
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
        naip_roi = self[:, ty:by+1, tx:bx+1]
        center_row, center_col = naip_roi.get_center()
        return naip_roi.get_lon_lat(row=center_row, col=center_col)

    def get_resolution(self) -> tuple[int, int]:
        """
        Get the resolution of the NAIP image in meters/pixel.
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

    def show_image(self, window_name: str) -> None:
        """
        Display the input image in a new window.
        :param window_name: str, title of the window displaying the image
        :return: None
        """
        cv2.imshow(window_name, self.get_bgr_naip())
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    def get_vegetation_by_hsv(self, lower_hue=25, upper_hue=100):
        lower = np.array([lower_hue, 30, 15])
        upper = np.array([upper_hue, 255, 255])
        hsv_img = cv2.cvtColor(self.get_bgr_naip(), cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_img, lower, upper)

    def calculate_ndvi(self) -> np.ndarray:
        """
        Calculate the normalized difference (NDVI) of the input NAIP image.
        :return: numpy array, NDVI results
        """
        naip_data = self._naip_img.values.astype(np.float64)
        return es.normalized_diff(naip_data[3], naip_data[0])

    def set_mask_color(self, mask_binary: np.ndarray,
                       pos_colors: tuple[int, int, int] = (0, 0, 255),
                       neg_colors: tuple[int, int, int] = (0, 0, 0)):
        """
        Apply a color specified by colors (B,G,R) to the NDVI mask (binary-classified)
        and return the colored mask.
        :param mask_binary: np.ndarray of data type "np.uint8", the binary-classified NDVI mask.
        :param pos_colors: tuple of int, specifying the (B, G, R) color applying to positive category pixels.
        :param neg_colors: tuple of int, specifying the (B, G, R) color applying to negative category pixels.
        :return: np.ndarray, colored NDVI mask of shape (height, width, color).
        """
        pos_res = self._set_category_color(mask_binary, pos_colors)

        mask_binary_invert = NAIPImagery._invert_binary_mask(mask_binary)
        neg_res = NAIPImagery._set_category_color(mask_binary_invert, neg_colors)

        return pos_res + neg_res

    @staticmethod
    def _invert_binary_mask(mask_binary):
        return ((mask_binary.astype(np.int32) - 1) * (-1)).astype(np.uint8)

    @staticmethod
    def _set_category_color(mask_binary, colors: tuple[int, int, int]):
        res = np.tile(mask_binary, (3, 1, 1)).transpose(1, 2, 0)
        res[:, :, 0] *= colors[0]
        res[:, :, 1] *= colors[1]
        res[:, :, 2] *= colors[2]
        return res

    def generate_vegetation_cover_map(self,
                                      mask_binary: np.ndarray,
                                      pos_colors: Tuple[int, int, int] = (0, 0, 255),
                                      neg_colors: Tuple[int, int, int] = (0, 0, 0)
                                      ) -> np.ndarray:
        # mask_gray = self.generate_vegetation_mask(ndvi_threshold, invert)
        mask_bgr = self.set_mask_color(mask_binary, pos_colors, neg_colors)
        return cv2.addWeighted(self.get_bgr_naip(), 1, mask_bgr, 0.25, 0)

    def generate_vegetation_mask(self,
                                 ndvi_threshold: float,
                                 invert: bool = False) -> np.ndarray:
        """
        Generate vegetation mask for the NAIP image.
        :param ndvi_threshold: float, NDVI threshold to classify pixels.
        :param invert: bool, set to invert the classification result.
                    Default is False, indicating 1 for green and 0 for non-green.
        :return: numpy array, vegetation mask representing if the pixels are green or not.
        """
        ndvi = self.calculate_ndvi()
        mask = np.zeros_like(ndvi, dtype=np.uint8)
        mask[ndvi >= ndvi_threshold] = 1
        mask[ndvi < ndvi_threshold] = 0
        if invert:
            mask = np.invert(mask.astype(np.bool_)).astype(np.uint8)
        return mask

    def integrate_vegetation_mask(self, mask):
        # naip_img_copy = copy.deepcopy(self.naip_img)
        naip_img_copy = self.naip_img.isel(band=0)
        naip_img_copy.values = mask
        return naip_img_copy

    def split_image(self, des_path: str, split_height: int, split_width: int) -> None:
        """
        Split the NAIP image into blocks of size (split_height, split_width). Blocks of smaller size are kept as-is.
        :param des_path: str, the output directory of the split images
        :param split_height: int, height of the blocks
        :param split_width: int, width of the blocks
        :return: None
        """
        util.create_directory(des_path)
        util.remove_all_files(des_path)

        input_image = self.get_bgr_naip()
        h, w = input_image.shape[:2]
        splits = [input_image[y:y + split_height, x:x + split_width]
                  for y in range(0, h, split_height)
                  for x in range(0, w, split_width)]
        n_digits = len(str(len(splits)))
        for i, split in enumerate(splits):
            if split is not None:
                filename = f"naip_split_rowSize{math.ceil(w / split_width)}_{str(i).zfill(n_digits)}.png"
                cv2.imwrite(os.path.join(des_path, filename), split)

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
    img_path = "../../image/m_4111118_nw_12_060_20210813.tif"


    naip_img = util.read_naip_image(img_path)
    naip = NAIPImagery(naip_img)
    print(naip.naip_img)
    # print(type(naip.naip_img.values))


    vegetation_mask = naip.generate_vegetation_mask(0.14)
    # vegetation_mask_bgr = naip.set_mask_color(vegetation_mask,
    #                                               pos_colors=(84, 163, 49),
    #                                               neg_colors=(185, 252, 247))
    vegetation_mask_integrated = naip.integrate_vegetation_mask(vegetation_mask)
    vegetation_mask_integrated.rio.to_raster("../../results/test_binary_mask_with_spatial.tif")
    # cv2.imwrite("../../results/test_bgr_mask1.png", vegetation_mask_bgr)
    # land_cover_map = naip.generate_vegetation_cover_map(vegetation_mask)
    # cv2.imwrite("../../results/test_land_cover1-5.png", land_cover_map)





