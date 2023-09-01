import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from naip_processor import NAIPProcessor


class ConnectedComponent:
    @staticmethod
    def generate_components(cc_input, connectivity):
        return cv2.connectedComponentsWithStats(cc_input.astype(np.uint8),
                                                connectivity,
                                                cv2.CV_32S)
    @staticmethod
    def summary_statistics(cc_output):
        (num_labels, labels, stats, centroids) = cc_output
        area_df = pd.DataFrame(stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]],
                               columns=['Width', 'Height', 'Area'])
        area_stats = area_df.describe()
        return area_stats

    @staticmethod
    def make_cc_mask(cc_output, width_filter=0, height_filter=0):
        (num_labels, labels, stats, centroids) = cc_output
        keep_count = 0
        component_masks = dict()  # key: label; value: mask value
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            keep_w = w >= width_filter
            keep_h = h >= height_filter

            if all((keep_w, keep_h)):
                keep_count += 1
                component_mask = (labels == i).astype("unit8") * 255
                component_masks[i] = component_mask
        return component_masks

    @staticmethod
    def visualize_cc(cc_output, naip_rgb):
        (num_labels, labels, stats, centroids) = cc_output
        width_filter = 3
        height_filter = 3
        keep_count = 0
        cc_masks = dict()  # key: label; value: numpy array of mask values
        frames = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (c_x, c_y) = centroids[i]

            keep_w = w >= width_filter
            keep_h = h >= height_filter

            if all((keep_w, keep_h)):
                keep_count += 1
                background = naip_rgb.copy()
                # cv2.rectangle(background, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(background, (int(c_x), int(c_y)), 2, (0, 0, 255), -1)

                cc_mask = (labels == i).astype("uint8") * 255
                cc_mask_eroded = ConnectedComponent.erode(cc_mask)
                cc_mask_rgb = cv2.cvtColor(cc_mask_eroded, cv2.COLOR_GRAY2BGR)
                cc_mask_rgb[:, :, 0][cc_mask_rgb[:, :, 0] == 255] = 0
                cc_mask_rgb[:, :, 2][cc_mask_rgb[:, :, 2] == 255] = 0
                # cc_mask_rgb[:, :, 0] = 0
                # cc_mask_rgb[:, :, 2] = 0
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

    @staticmethod
    def generate_rondom_locations(naip_img, component_masks):
        cc_random_location = np.zeros((len(component_masks), 3))
        for i, (key, value) in enumerate(component_masks.items()):
            samples = np.where(value == 255)
            j = np.random.randint(0, len(samples[0]))  # Randomly select a location where the mask value is 255
            cc_random_location[i, 0] = key
            lon = naip_img[0, samples[0][j], samples[1][j]].x.values  # samples[0]: y or height; samples[1]: x or width
            lat = naip_img[0, samples[0][j], samples[1][j]].y.values
            cc_random_location[i, 1] = lon
            cc_random_location[i, 2] = lat

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
    cc_output = ConnectedComponent.generate_components(ndvi_classified, 8)

    area_stats = ConnectedComponent.summary_statistics(cc_output)
    # print(area_stats.round(2))

    cc_masks, frames = ConnectedComponent.visualize_cc(cc_output, naip_rgb)
    print(len(cc_masks))
    ConnectedComponent.make_video(frames, "./results/cc_video02.avi")

