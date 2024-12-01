import os

import cv2
import numpy as np
from abc import ABC, abstractmethod

from utility.mask_factory import MaskCreator


class ConfusionMatrix:
    def __init__(self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

    @property
    def tp(self):
        return self._tp

    @tp.setter
    def tp(self, value):
        self._tp = value

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, value):
        self._fp = value

    @property
    def tn(self):
        return self._tn

    @tn.setter
    def tn(self, value):
        self._tn = value

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        self._fn = value

    def get_kappa(self):
        try:
            numerator = self._tp * self._tn - self._fp * self._fn
            denominator = ((self._tp + self._fp) * (self._fp + self._tn) +
                       (self._tp + self._fn) * (self._fn + self._tn))
            return 2.0 * numerator / denominator
        except ZeroDivisionError as err:
            print(f"{err}: Failed to calculate kappa")
            return -1

    def get_accuracy(self):
        try:
            return 1.0 * (self._tp + self._tn) / (self._tp + self._fp + self._tn + self._fn)
        except ZeroDivisionError as err:
            print(f"{err}: Failed to calculate accuracy")
            return -1

    def get_confusion_matrix(self):
        return {"tp": self._tp,
                "fp": self._fp,
                "tn": self._tn,
                "fn": self._fn,
                "accuracy": self.get_accuracy(),
                "kappa": self.get_kappa()}

    def compute_on_single_sample(self, actual_mask: np.ndarray, predicted_mask: np.ndarray) -> None:
        tp_matrix = np.logical_and(actual_mask, predicted_mask)
        tp = int(np.sum(tp_matrix))
        self._tp += tp

        # Accumulate TP and TN
        tn_matrix = np.logical_and(np.logical_not(actual_mask), np.logical_not(predicted_mask))
        tn = int(np.sum(tn_matrix))
        self._tn += tn

        # Accumulate FP and FN
        n_fp = int(np.sum(predicted_mask) / 255 - tp)
        self._fp += n_fp

        n_fn = tp_matrix.size - tp - tn - n_fp
        self._fn += n_fn

    def compute_on_batch_samples(self,
                                 actual_mask_path: str,
                                 predicted_mask_path: str,
                                 mask_creator: MaskCreator) -> None:
        actual_mask_fp = os.scandir(actual_mask_path)
        predicted_mask_fp = os.scandir(predicted_mask_path)

        for actual_mask_obj, predicted_mask_obj in zip(actual_mask_fp, predicted_mask_fp):
            actual_full_path = os.path.join(actual_mask_path, actual_mask_obj.name)
            predicted_full_path = os.path.join(predicted_mask_path, predicted_mask_obj.name)
            actual_mask, predicted_mask = mask_creator.create(actual_full_path, predicted_full_path)
            self.compute_on_single_sample(actual_mask, predicted_mask)

