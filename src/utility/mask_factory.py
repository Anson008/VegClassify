from abc import ABC, abstractmethod

import cv2
import numpy as np


class MaskCreator(ABC):
    @abstractmethod
    def factory_method(self):
        pass

    def create(self, actual_mask_path: str, predicted_mask_path: str):
        mask = self.factory_method()
        return mask.generate(actual_mask_path, predicted_mask_path)


class FullMaskCreator(MaskCreator):
    def factory_method(self):
        return FullMask()

class RandomSampledMaskCreator(MaskCreator):
    def __init__(self, sample_size):
        self._sample_size = sample_size

    def factory_method(self):
        return RandomSampledMask(self._sample_size)


class Mask(ABC):
    def __init__(self):
        self._height = 0
        self._width = 0

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @abstractmethod
    def generate(self, actual_mask_path: str, predicted_mask_path: str):
        pass

class FullMask(Mask):
    def generate(self, actual_mask_path: str, predicted_mask_path: str):
        gt_mask = cv2.imread(actual_mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

        self._height = gt_mask.shape[0]
        self._width = gt_mask.shape[1]
        return gt_mask, predicted_mask

class RandomSampledMask(Mask):
    def __init__(self, sample_size=20):
        super().__init__()
        self._index_array = np.zeros(1)
        self._sample_size = sample_size

    def generate(self, actual_mask_path: str, predicted_mask_path: str):
        gt_mask = cv2.imread(actual_mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

        self._height = gt_mask.shape[0]
        self._width = gt_mask.shape[1]
        self._index_array = np.arange(self._height * self._width)

        random_idx = np.random.choice(self._index_array, self._sample_size, replace=False)
        random_idx = np.unravel_index(random_idx, gt_mask.shape)

        gt_mask_sample = gt_mask[random_idx[0], random_idx[1]]
        predicted_mask_sample = predicted_mask[random_idx[0], random_idx[1]]

        return gt_mask_sample, predicted_mask_sample