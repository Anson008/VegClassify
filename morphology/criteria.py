import cv2
from abc import ABC, abstractmethod
from morphology.connected_components import ConnectedComponents, CV2ConnectedComponentsGenerator
from ndvi.naip_processor import NAIPProcessor
import numpy as np


class CriteriaBase(ABC):
    param_dict = {"width": cv2.CC_STAT_WIDTH,
                  "height": cv2.CC_STAT_HEIGHT,
                  "area": cv2.CC_STAT_AREA}

    def __init__(self, param_name, target):
        self._param_name = param_name
        self._target = target

    @abstractmethod
    def apply(self, connected_component):
        pass


class GreaterThanOrEqualToCriteria(CriteriaBase):
    def apply(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        num_labels = 0
        labels = []
        stats = []
        centroids = []
        for i in range(connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param >= self._target:
                label = connected_component.labels[i, :, :]
                label[label == i + 1] = i + 1
                labels.append(label)
                stats.append(connected_component.stats[i, :])
                centroids.append(connected_component.centroids[i, :])
                num_labels += 1
        labels = np.array(labels)
        stats = np.array(stats)
        centroids = np.array(centroids)
        return ConnectedComponents((num_labels, labels, stats, centroids))


class GreaterThanCriteria(CriteriaBase):
    def apply(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        num_labels = 0
        labels = []
        stats = []
        centroids = []
        for i in range(connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param > self._target:
                label = connected_component.labels[i, :, :]
                label[label == i + 1] = i + 1
                labels.append(label)
                stats.append(connected_component.stats[i])
                centroids.append(connected_component.centroids[i])
                num_labels += 1
        labels = np.array(labels)
        stats = np.array(stats)
        centroids = np.array(centroids)
        return ConnectedComponents((num_labels, labels, stats, centroids))


class LessThanOrEqualToCriteria(CriteriaBase):
    def apply(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        num_labels = 0
        labels = []
        stats = []
        centroids = []
        for i in range(connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param <= self._target:
                label = connected_component.labels[i, :, :]
                label[label == i + 1] = i + 1
                labels.append(label)
                stats.append(connected_component.stats[i])
                centroids.append(connected_component.centroids[i])
                num_labels += 1
        labels = np.array(labels)
        stats = np.array(stats)
        centroids = np.array(centroids)
        return ConnectedComponents((num_labels, labels, stats, centroids))


class LessThanCriteria(CriteriaBase):
    def apply(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        num_labels = 0
        labels = []
        stats = []
        centroids = []
        for i in range(connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param < self._target:
                label = connected_component.labels[i, :, :]
                label[label == i + 1] = i + 1
                labels.append(label)
                stats.append(connected_component.stats[i])
                centroids.append(connected_component.centroids[i])
                num_labels += 1
        labels = np.array(labels)
        stats = np.array(stats)
        centroids = np.array(centroids)
        return ConnectedComponents((num_labels, labels, stats, centroids))


class EqualToCriteria(CriteriaBase):
    def apply(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        num_labels = 0
        labels = []
        stats = []
        centroids = []
        for i in range(connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param == self._target:
                label = connected_component.labels[i, :, :]
                label[label == i + 1] = i + 1
                labels.append(label)
                stats.append(connected_component.stats[i])
                centroids.append(connected_component.centroids[i])
                num_labels += 1
        labels = np.array(labels)
        stats = np.array(stats)
        centroids = np.array(centroids)
        return ConnectedComponents((num_labels, labels, stats, centroids))


class NotEqualToCriteria(CriteriaBase):
    def apply(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        num_labels = 0
        labels = []
        stats = []
        centroids = []
        for i in range(connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param != self._target:
                label = connected_component.labels[i, :, :]
                label[label == i + 1] = i + 1
                labels.append(label)
                stats.append(connected_component.stats[i])
                centroids.append(connected_component.centroids[i])
                num_labels += 1
        labels = np.array(labels)
        stats = np.array(stats)
        centroids = np.array(centroids)
        return ConnectedComponents((num_labels, labels, stats, centroids))


class AndCriteria:

    def __init__(self, criteria1, criteria2):
        self._criteria1 = criteria1
        self._criteria2 = criteria2

    def apply_criteria(self, cc_result):
        temp = self._criteria1.apply(cc_result)
        return self._criteria2.apply(temp)


if __name__ == "__main__":
    img_path = "../image/m_4111118_nw_12_060_20210813_Clip.tif"
    naip = NAIPProcessor(img_path)
    naip_rgb = naip.get_rgb_naip()
    naip_reprojected = naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)
    ndvi_classified = NAIPProcessor.classify(ndvi, 0.11)
    cc_generator = CV2ConnectedComponentsGenerator(ndvi_classified, 8)
    cc_res = cc_generator.generate()
    cc_obj = ConnectedComponents(cc_res)
    area_stats = cc_obj.summary_statistics()
    # print(area_stats.round(2))

    not_less_than = GreaterThanOrEqualToCriteria("area", 100)
    cc_area_gteq_100 = not_less_than.apply(cc_obj)
    filtered_stats = cc_area_gteq_100.summary_statistics()
    print(filtered_stats)

    criteria1 = GreaterThanOrEqualToCriteria("height", 10)
    criteria2 = LessThanOrEqualToCriteria("width", 30)
    criteria3 = AndCriteria(criteria1, criteria2)
    res3 = criteria3.apply_criteria(cc_obj)
    print(res3.summary_statistics())