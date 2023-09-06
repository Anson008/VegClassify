import cv2
import numpy as np
import pandas as pd
import math
import json
import matplotlib.pyplot as plt
from naip_processor import NAIPProcessor


class ConnectedComponents:
    def __init__(self, cc_result):
        self.__num_labels = cc_result[0]
        self.__labels = cc_result[1]
        self.__stats = cc_result[2]
        self.__centroids = cc_result[3]

    @property
    def num_labels(self):
        return self.__num_labels

    @property
    def labels(self):
        return self.__labels

    @property
    def stats(self):
        return self.__stats

    @property
    def centroids(self):
        return self.__centroids


class ConnectedComponentsAnalyser:
    def __init__(self, cc_input, connectivity=8):
        self._cc_input = cc_input
        self._connectivity = connectivity
        output = cv2.connectedComponentsWithStats(self._cc_input.astype(np.uint8),
                                                  self._connectivity,
                                                  cv2.CV_32S)
        self._cc_result = ConnectedComponents(output)

    @property
    def cc_input(self):
        return self._cc_input

    @cc_input.setter
    def cc_input(self, cc_input):
        self._cc_input = cc_input

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value):
        self._connectivity = value

    @property
    def cc_result(self):
        return self._cc_result

    def update_cc_result(self):
        output = cv2.connectedComponentsWithStats(self._cc_input.astype(np.uint8),
                                                  self._connectivity,
                                                  cv2.CV_32S)
        self._cc_result = ConnectedComponents(output)

    @staticmethod
    def generate_components(cc_input, connectivity):
        return cv2.connectedComponentsWithStats(cc_input.astype(np.uint8),
                                                connectivity,
                                                cv2.CV_32S)

    def summary_statistics(self):
        dimension_df = pd.DataFrame(self._cc_result.stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]],
                                    columns=['Width', 'Height', 'Area'])
        res = dimension_df.describe()
        return res

    def make_cc_mask(self, width_filter=0, height_filter=0):
        keep_count = 0
        component_masks = dict()  # key: label; value: mask value
        for i in range(1, self._cc_result.num_labels):
            w = self._cc_result.stats[i, cv2.CC_STAT_WIDTH]
            h = self._cc_result.stats[i, cv2.CC_STAT_HEIGHT]

            keep_w = w >= width_filter
            keep_h = h >= height_filter

            if all((keep_w, keep_h)):
                keep_count += 1
                component_mask = (self._cc_result.labels == i).astype("unit8") * 255
                component_masks[i] = component_mask
        return component_masks

    def visualize_cc(self, naip_rgb):

        width_filter = 3
        height_filter = 3
        keep_count = 0
        cc_masks = dict()  # key: label; value: numpy array of mask values
        frames = []
        for i in range(1, self._cc_result.num_labels):
            w = self._cc_result.stats[i, cv2.CC_STAT_WIDTH]
            h = self._cc_result.stats[i, cv2.CC_STAT_HEIGHT]
            area = self._cc_result.stats[i, cv2.CC_STAT_AREA]
            (c_x, c_y) = self._cc_result.centroids[i]

            keep_w = w >= width_filter
            keep_h = h >= height_filter

            if all((keep_w, keep_h)):
                keep_count += 1
                background = naip_rgb.copy()
                cv2.circle(background, (int(c_x), int(c_y)), 2, (0, 0, 255), -1)

                cc_mask = (self._cc_result.labels == i).astype("uint8") * 255
                cc_mask_eroded = ConnectedComponentsAnalyser.erode(cc_mask)
                cc_mask_rgb = cv2.cvtColor(cc_mask_eroded, cv2.COLOR_GRAY2BGR)
                cc_mask_rgb[:, :, 0][cc_mask_rgb[:, :, 0] == 255] = 0  # Set blue channel to 0
                cc_mask_rgb[:, :, 2][cc_mask_rgb[:, :, 2] == 255] = 0  # Set red channel to 0
                # cv2.imshow("Test", cc_mask_rgb)
                # cv2.waitKey(0)
                # cv2.destroyWindow("Test")
                cc_masks[i] = cc_mask_eroded

                # naip_b, naip_g, naip_r = cv2.split(naip_rgb)
                # naip_alpha = np.ones(naip_b.shape, dtype=naip_b.dtype)
                frame = cv2.addWeighted(background, 1, cc_mask_rgb, 0.5, 0)
                frames.append(frame)

        return cc_masks, frames

    @staticmethod
    def erode(cc_mask, is_invert=False):
        kernel = np.ones((3, 3), np.uint8)
        if is_invert:
            cc_mask = cv2.bitwise_not(cc_mask)
        erosion = cv2.erode(cc_mask, kernel, iterations=1)
        return erosion

    @staticmethod
    def make_video(frames, path):
        if len(frames) > 0:
            size = (frames[0].shape[1], frames[0].shape[0])
            video_out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 1, size)

            for i in range(len(frames)):
                video_out.write(frames[i])
            video_out.release()

    def generate_rondom_locations(self, naip_img, sample_rate):
        cc_random_location = dict()
        for i in range(1, self._cc_result.num_labels):
            samples = np.where(self._cc_result.labels == i)

            # Randomly select a location where the mask value is 255
            num_pixels = samples[0].shape[0]
            sample_index = np.random.choice(np.arange(num_pixels), size=math.ceil(num_pixels * sample_rate))
            for j in list(sample_index):
                lon = naip_img[0, samples[0][j], samples[1][j]].x.values.item()  # samples[0]: y or height; samples[1]: x or width
                lat = naip_img[0, samples[0][j], samples[1][j]].y.values.item()
                # print("lat type:", type(lat))
                cc_random_location.setdefault(i, []).append((lon, lat))

        return cc_random_location

    @staticmethod
    def gather_cc_info(cc_output):
        (num_labels, labels, stats, centroids) = cc_output
        cc_labels = np.arange(num_labels).reshape((num_labels, 1))
        cc_info = np.hstack((cc_labels, stats, centroids))
        return cc_info

    @staticmethod
    def append_random_locations_to_cc_info(naip_img, cc_info, cc_random_location):
        pass


if __name__ == "__main__":
    img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
    naip = NAIPProcessor(img_path)
    naip_rgb = naip.get_rgb_naip()
    naip_reprojected = naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)
    ndvi_classified = NAIPProcessor.classify(ndvi, 0.2)
    cc_analyser = ConnectedComponentsAnalyser(ndvi_classified, 8)

    area_stats = cc_analyser.summary_statistics()
    # print(area_stats.round(2))

    random_locations = cc_analyser.generate_rondom_locations(naip_reprojected, 0.01)
    with open("./results/random_locations.txt", "w") as file_handler:
        file_handler.write(json.dumps(random_locations))
    # for key, value in random_locations.items():
    #     print(f"Random locations for the {key}th component:\n")
    #     for (lon, lat) in value:
    #         print(f"Lon: {lon:.6f}; Lat: {lat:.6f}\n")

    # cc_masks, frames = cc_analyser.visualize_cc(naip_rgb)
    # ConnectedComponentsAnalyser.make_video(frames, "./results/cc_video_eroded.avi")

