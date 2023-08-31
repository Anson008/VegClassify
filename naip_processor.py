import cv2
import rioxarray as rxr
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NAIPProcessor:

    def __init__(self, img_path=None):
        """
        :param img_path: str, path of the input NAIP image
        """
        self._naip_img = None
        if img_path:
            self._naip_img = rxr.open_rasterio(img_path)

    @property
    def naip_img(self):
        """
        :return: xarray.DataArray, raw NAIP image
        """
        return self._naip_img

    @naip_img.setter
    def naip_img(self, img_path):
        """
        :param img_path: str, path of NAIP image
        """
        self._naip_img = rxr.open_rasterio(img_path)

    def reproject(self, dst_crs):
        """
        :param dst_crs: str, OGC WKT string or Proj.4 string
        :return: xarray.DataArray, reprojected NAIP image
        """
        # Reproject NAIP to dst_crs with the same shape
        _, height, width = self.naip_img.shape
        return self._naip_img.rio.reproject(dst_crs, shape=(height, width))

    def get_rgb_naip(self):
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
    def classify(ndvi, threshold):
        """
        :param ndvi: numpy array, NDVI values for an image
        :param threshold: float, the value to distinguish green space and non-green space
        :return: numpy array, classified pixel values of the image
        """
        ndvi[ndvi >= threshold] = 1
        ndvi[ndvi < threshold] = 0
        return ndvi

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


if __name__ == "__main__":
    img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
    naip = NAIPProcessor()
    naip.naip_img = img_path
    naip_rgb = naip.get_rgb_naip()
    NAIPProcessor.show_image("RGB", naip_rgb)
    naip_reprojected = naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)
    ndvi_classified = NAIPProcessor.classify(ndvi, 0.2)
    NAIPProcessor.plot_bands(ndvi_classified, "PiYG", title="NDVI")