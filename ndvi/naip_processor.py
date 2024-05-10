import cv2
import rioxarray as rxr
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import matplotlib.pyplot as plt
import os
import util
import math


class NAIPProcessor:
    NAIP_SPLIT_DIR = "../cache/naip_split/"

    def __init__(self, naip_img):
        """
        :param img_path: str, path of the input NAIP image
        """
        self._naip_img = naip_img

    @property
    def naip_img(self):
        """
        :return: xarray.DataArray, raw NAIP image
        """
        return self._naip_img

    @naip_img.setter
    def naip_img(self, new_naip_img):
        """
        :param: xarray.DataArray, a new NAIP image
        """
        self._naip_img = new_naip_img

    def get_center(self):
        height, width = self.naip_img.shape[1:]
        return (height - 1) // 2, (width - 1) // 2

    def reproject(self, dst_crs="EPSG:4326"):
        """
        :param dst_crs: str, OGC WKT string or Proj.4 string
        :return: xarray.DataArray, reprojected NAIP image
        """
        # Reproject NAIP to dst_crs with the same shape
        _, height, width = self.naip_img.shape
        return self._naip_img.rio.reproject(dst_crs, shape=(height, width))

    def get_lon_lat(self, dst_crs="EPSG:4326", row=0, col=0):
        naip = self.reproject(dst_crs)
        return naip[0, row, col].x.values.item(), naip[0, row, col].y.values.item()

    def get_resolution(self):
        return self.naip_img.rio.resolution()

    def get_bgr_naip(self):
        """
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
    def show_image(window_name, img):
        """
        :param window_name: str, title of the window displaying the image
        :param img: numpy array, image to display
        :return: None
        """
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    @staticmethod
    def calculate_ndvi(naip_img):
        """
        :param naip_img: xarray.DataArray, NAIP image to perform the calculation on
        :return: numpy array, NDVI results
        """
        naip_data = naip_img.values.astype(np.float64)
        return es.normalized_diff(naip_data[3], naip_data[0])

    @staticmethod
    def classify(ndvi, threshold, invert=False):
        """
        :param ndvi: numpy array, NDVI values for an image
        :param threshold: float, the value to distinguish green space and non-green space
        :return: numpy array, classified pixel values of the image
        """
        classified_ndvi = np.zeros_like(ndvi, dtype=np.uint8)
        classified_ndvi[ndvi >= threshold] = 1
        classified_ndvi[ndvi < threshold] = 0
        if invert:
            classified_ndvi = np.invert(classified_ndvi.astype(np.bool_)).astype(np.uint8)
        classified_ndvi[classified_ndvi != 0] = 255
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

    def split_image(self, split_height, split_width):
        util.create_directory(self.NAIP_SPLIT_DIR)
        util.remove_all_files(self.NAIP_SPLIT_DIR)
        input_image = self.get_bgr_naip()
        h, w = input_image.shape[:2]
        splits = [input_image[y:y + split_height, x:x + split_width]
                  for y in range(0, h, split_height)
                  for x in range(0, w, split_width)]
        n_digits = len(str(len(splits)))
        for i, split in enumerate(splits):
            if split is not None:
                filename = f"naip_split_rowSize{math.ceil(w / split_width)}_{str(i).zfill(n_digits)}.png"
                cv2.imwrite(os.path.join(self.NAIP_SPLIT_DIR, filename), split)


if __name__ == "__main__":
    img_path = "../image/m_4111118_nw_12_060_20210813.tif"

    naip_img = util.read_naip_image(img_path)
    naip = NAIPProcessor(naip_img)
    naip.split_image(1024, 512)







