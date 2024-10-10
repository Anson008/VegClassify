import numpy as np
from abc import ABC, abstractmethod


class SampleMethod(ABC):
    @abstractmethod
    def sample(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        pass

    @staticmethod
    def _make_diagonal_coordinates(top_left_x: np.ndarray,
                                   top_left_y: np.ndarray,
                                   bottom_right_x: np.ndarray,
                                   bottom_right_y: np.ndarray) -> np.ndarray:
        """
        Assemble the coordinates of the top-left and bottom-right corners.
        :param top_left_x: numpy array, x coordinates of the top left corner
        :param top_left_y: numpy array, y coordinates of the top left corner
        :param bottom_right_x: numpy array, x coordinates of the bottom right corner
        :param bottom_right_y: numpy array, y coordinates of the bottom right corner
        :return: numpy array of shape (n_samples, 4). Each row is the coordinates of top-left and bottom-right corners.
        """
        top_left_xy = np.array(np.meshgrid(top_left_x, top_left_y)).T.reshape(-1, 2)
        bottom_right_xy = np.array(np.meshgrid(bottom_right_x, bottom_right_y)).T.reshape(-1, 2)
        diagonal_xy = np.concatenate((top_left_xy, bottom_right_xy), axis=1)
        return diagonal_xy


class RandomSampleMethod(SampleMethod):
    def __init__(self, seed: int | None = None):
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int | None):
        self._seed = value

    def sample(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        pass


class RandomSampleBySize(RandomSampleMethod):
    def sample(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        """
        Generate a naip slice at a random location and return the coordinates of
        the top-left and bottom-right of the slice.
        :param naip_size: tuple of int, the shape of NAIP (height, width).
        :param sample_shape: tuple of int, the block size (height, width) to split the NAIP into a grid.
        :return: np.ndarray of shape (n_samples, 4).
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y]. The block includes the bottom_right coordinates.
        """
        naip_h = naip_size[0]
        naip_w = naip_size[1]

        sample_h = sample_shape[0]
        sample_w = sample_shape[1]

        rng = np.random.default_rng(self._seed)
        top_left_x = rng.integers(0, naip_w - sample_w, dtype=np.int32)
        top_left_y = rng.integers(0, naip_h - sample_h, dtype=np.int32)
        bottom_right_x = top_left_x + sample_w
        bottom_right_y = top_left_y + sample_h

        return self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)


class RandomSampleByNumberPerDimension(RandomSampleMethod):
    def __init__(self, n_samples_x: int, n_samples_y: int):
        super().__init__()
        self._n_samples_x = n_samples_x
        self._n_samples_y = n_samples_y

    def sample(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        naip_h = naip_size[0]
        naip_w = naip_size[1]

        max_block_h = sample_shape[0]
        max_block_w = sample_shape[1]

        sample_w = min(int(naip_w / self._n_samples_x), max_block_w)
        sample_h = min(int(naip_h / self._n_samples_y), max_block_h)

        rng = np.random.default_rng(self._seed)
        top_left_x = rng.integers(0, naip_w - sample_w + 1, self._n_samples_x, dtype=np.int32)
        top_left_y = rng.integers(0, naip_h - sample_h + 1, self._n_samples_y, dtype=np.int32)
        bottom_right_x = top_left_x + sample_w
        bottom_right_y = top_left_y + sample_h

        return self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)


class GridSample(SampleMethod):
    def sample(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        """
        Divide the NAIP image into grid samples and return the coordinates of
        the top-left and bottom-right of the grid samples.
        :param naip_size: tuple of int, the shape of NAIP (height, width).
        :param sample_shape: tuple of int, the block size (height, width) to split the NAIP into a grid.
        :return: np.ndarray of shape (n_samples, 4).
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y]. The block includes the bottom_right coordinates.
        """
        naip_h = naip_size[0]
        naip_w = naip_size[1]

        block_h = sample_shape[0] if naip_h >= sample_shape[0] else naip_h
        block_w = sample_shape[1] if naip_w >= sample_shape[1] else naip_w

        top_left_x = np.arange(0, naip_w - block_w + 1, block_w, dtype=np.int32)
        top_left_y = np.arange(0, naip_h - block_h + 1, block_h, dtype=np.int32)
        bottom_right_x = top_left_x + block_w
        bottom_right_y = top_left_y + block_h

        return self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
