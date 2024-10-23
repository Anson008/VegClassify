import cv2
from abc import ABC, abstractmethod
from src.morphology.connected_components import ConnectedComponents
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