import cv2
from abc import ABC, abstractmethod


class CriteriaBase(ABC):
    param_dict = {"width": cv2.CC_STAT_WIDTH,
                  "height": cv2.CC_STAT_HEIGHT,
                  "area": cv2.CC_STAT_AREA}

    def __init__(self, param_name, target):
        self._param_name = param_name
        self._target = target

    @abstractmethod
    def filter(self, connected_component):
        pass


class GreaterThanOrEqualToCriteria(CriteriaBase):

    def filter(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        res = []
        for i in range(1, connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param >= self._target:

                res.append((connected_component.labels[i], connected_component.centroids[i]))
        return res


class GreaterThanCriteria(CriteriaBase):
    def filter(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        res = []
        for i in range(1, connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param > self._target:
                res.append((connected_component.labels[i], connected_component.centroids[i]))
        return res


class LessThanOrEqualToCriteria(CriteriaBase):

    def filter(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        res = []
        for i in range(1, connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param <= self._target:
                res.append((connected_component.labels[i], connected_component.centroids[i]))
        return res


class LessThanCriteria(CriteriaBase):
    def filter(self, connected_component):
        cv2_param = self.param_dict[self._param_name]
        res = []
        for i in range(1, connected_component.num_labels):
            param = connected_component.stats[i, cv2_param]
            if param < self._target:
                res.append((connected_component.labels[i], connected_component.centroids[i]))
        return res


class AndCriteria:

    def __init__(self, criteria1, criteria2):
        self._criteria1 = criteria1
        self._criteria2 = criteria2

    def apply_criteria(self, cc_result):
        res_1 = self._criteria1.filter(cc_result)
        return self._criteria2


class OrCriteria:
    def make_criteria(self):
        pass

