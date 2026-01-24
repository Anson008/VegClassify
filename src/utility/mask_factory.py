from abc import ABC, abstractmethod
import cv2
import numpy as np


class MaskCreator(ABC):
    def __init__(self, height, width):
        self._height = height
        self._width = width

    @abstractmethod
    def factory_method(self):
        pass

    def create(self, actual_mask_path: str, predicted_mask_path: str):
        mask = self.factory_method()
        return mask.generate(actual_mask_path, predicted_mask_path)


class FullMaskCreator(MaskCreator):
    def factory_method(self):
        return FullMask(self._height, self._width)

class RandomSampledMaskCreator(MaskCreator):
    def __init__(self, height, width, sample_size, seed):
        super().__init__(height, width)
        self._sample_size = sample_size
        self._seed = seed

    def factory_method(self):
        return RandomSampledMask(self._height, self._width, self._sample_size, self._seed)


class Mask(ABC):
    def __init__(self, height, width):
        self._height = height
        self._width = width

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

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
    def __init__(self, height, width, sample_size=20, seed=None):
        super().__init__(height, width)
        self._index_array = np.arange(self._height * self._width)
        self._sample_size = sample_size
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    def generate(self, actual_mask_path: str, predicted_mask_path: str):
        gt_mask = cv2.imread(actual_mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

        rng = np.random.default_rng(self._seed)

        random_idx = rng.choice(self._index_array, self._sample_size, replace=False)
        random_idx = np.unravel_index(random_idx, gt_mask.shape)

        gt_mask_sample = gt_mask[random_idx[0], random_idx[1]]
        predicted_mask_sample = predicted_mask[random_idx[0], random_idx[1]]

        return gt_mask_sample, predicted_mask_sample