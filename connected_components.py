import cv2
import numpy as np
import pandas as pd
import math
import json
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

    def to_tuple(self):
        return self.__num_labels, self.__labels, self.__stats, self.__centroids

    def summary_statistics(self):
        dimension_df = pd.DataFrame(self.__stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]],
                                    columns=['Width', 'Height', 'Area'])
        res = dimension_df.describe()
        return res

    def apply_filters(self, filters):
        res = self
        for filter in filters:
            res = filter.apply(res)
        return res


class CV2ConnectedComponentsGenerator:
    def __init__(self, cc_input, connectivity=8):
        self._cc_input = cc_input
        self._connectivity = connectivity

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

    def generate(self):
        return cv2.connectedComponentsWithStats(self._cc_input.astype(np.uint8),
                                                self._connectivity,
                                                cv2.CV_32S)


class ConnectedComponentsProcessor:

    @staticmethod
    def make_mask(cc_object):
        component_masks = dict()  # key: label; value: mask value
        combined_mask = np.zeros(cc_object.labels.shape, dtype='bool_')
        for i in range(1, cc_object.num_labels):
            component_mask = (cc_object.labels == i)
            combined_mask = np.ma.mask_or(combined_mask, component_mask)
            component_masks[i] = component_mask.astype("uint8") * 255
        return component_masks, combined_mask.astype("uint8") * 255

    @staticmethod
    def overlap_on_map(combined_mask, naip_rgb, mask_color, erosion=False):
        frame = naip_rgb.copy()
        if erosion:
            combined_mask = ConnectedComponentsProcessor.erode(combined_mask)
        mask_rgb = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        mask_rgb = ConnectedComponentsProcessor.__set_mask_color(mask_rgb, mask_color)
        frame = cv2.addWeighted(frame, 1, mask_rgb, 0.5, 0)
        return frame

    @staticmethod
    def __set_mask_color(mask, color):
        if color == "red":
            mask[:, :, 0][mask[:, :, 0] == 255] = 0  # Set blue channel to 0
            mask[:, :, 1][mask[:, :, 1] == 255] = 0  # Set green channel to 0
        elif color == "blue":
            mask[:, :, 1][mask[:, :, 1] == 255] = 0
            mask[:, :, 2][mask[:, :, 2] == 255] = 0  # Set red channel to 0
        elif color == "green":
            mask[:, :, 1][mask[:, :, 0] == 255] = 0
            mask[:, :, 2][mask[:, :, 2] == 255] = 0  # Set red channel to 0
        return mask

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

    @staticmethod
    def mark_cc_on_map(cc_object, naip_rgb, box_on_area_larger_than=float('inf')):
        frame = naip_rgb.copy()
        for i in range(1, cc_object.num_labels):
            x = cc_object.stats[i, cv2.CC_STAT_LEFT]
            y = cc_object.stats[i, cv2.CC_STAT_TOP]
            w = cc_object.stats[i, cv2.CC_STAT_WIDTH]
            h = cc_object.stats[i, cv2.CC_STAT_HEIGHT]
            area = cc_object.stats[i, cv2.CC_STAT_AREA]
            (c_x, c_y) = cc_object.centroids[i]

            if area >= box_on_area_larger_than:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(frame, (int(c_x), int(c_y)), 1, (0, 0, 255), 0)
        return frame

    @staticmethod
    def generate_random_locations(cc_object, naip_img, sample_rate=0.001):
        """

        :param cc_object: ConnectedComponents, cc object to be sampled
        :param naip_img: rioxarray, naip map reprojected to EPSG:4326 coordinate system
        :param sample_rate: float, sample rate
        :return: dict, key: cc index, value: (lat, lon) of each sample point
        """
        random_geo_location = dict()

        for i in range(1, cc_object.num_labels):
            samples = np.where(cc_object.labels == i)

            # Find total number of pixels in current component
            num_pixels = samples[0].shape[0]

            # Randomly select a number of points in current component
            sample_index = np.random.choice(np.arange(num_pixels), size=math.ceil(num_pixels * sample_rate))

            # Find the geological location of sample points
            for j in list(sample_index):
                # samples[0]: y or height; samples[1]: x or width
                lon = naip_img[0, samples[0][j], samples[1][j]].x.values.item()
                lat = naip_img[0, samples[0][j], samples[1][j]].y.values.item()
                random_geo_location.setdefault(i, []).append((lon, lat))

        return random_geo_location

    @staticmethod
    def save_to_json(data, path):
        with open(path, "w") as fh:
            fh.write(json.dumps(data, indent=2))

    @staticmethod
    def save_image(img, path):
        cv2.imwrite(path, img)

    @staticmethod
    def gather_cc_info(cc_object):
        cc_index = np.arange(cc_object.num_labels).reshape((cc_object.num_labels, 1))
        cc_info = np.hstack((cc_index, cc_object.stats, cc_object.centroids))
        return cc_info


if __name__ == "__main__":
    img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
    naip = NAIPProcessor(img_path)
    naip_rgb = naip.get_rgb_naip()
    naip_reprojected = naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)
    ndvi_classified = NAIPProcessor.classify(ndvi, 0.11, invert=True)
    cv2_cc_generator = CV2ConnectedComponentsGenerator(ndvi_classified, 8)
    cc_results = cv2_cc_generator.generate()

    cc_object = ConnectedComponents(cc_results)
    # area_stats = cc_object.summary_statistics()
    # print(area_stats.round(2))

    geo_loc = ConnectedComponentsProcessor.generate_random_locations(cc_object, naip_reprojected)
    ConnectedComponentsProcessor.save_to_json(geo_loc, "./results/random_geo_locations.json")

    # cc_masks, frames = cc_object.visualize_cc(naip_rgb)
    # masks, combined_mask = ConnectedComponentsProcessor.make_mask(cc_object)
    # # print(len(masks))
    # frames = ConnectedComponentsProcessor.overlap_on_map(combined_mask, naip_rgb, "red", True)
    # # CV2ConnectedComponentsGenerator.make_video(frames, "./results/cc_video_area_test.avi")
    # frames = ConnectedComponentsProcessor.mark_cc_on_map(cc_object, naip_rgb, 25)
    # cv2.imwrite("./results/overlap_test8.png", frames)

