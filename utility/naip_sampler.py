import numpy as np


class NaipSampler:
    def __init__(self, naip_h, naip_w):
        self._naip_h = naip_h
        self._naip_w = naip_w
        self._samples = None

    @property
    def naip_h(self):
        return self._naip_h

    @naip_h.setter
    def naip_h(self, naip_h):
        self._naip_h = naip_h

    @property
    def naip_w(self):
        return self._naip_w

    @naip_w.setter
    def naip_w(self, naip_w):
        self._naip_w = naip_w

    @property
    def samples(self):
        return self._samples

    def get_num_of_samples(self):
        return self._samples.shape[0]

    def get_random_naip_imagery_samples(self, n_samples_xy=(2, 2), seed=None):
        s_w = min(int(self._naip_w / n_samples_xy[0]), 256)
        s_h = min(int(self._naip_h / n_samples_xy[1]), 512)

        rng = np.random.default_rng(seed)
        top_left_x = rng.integers(0, self._naip_w - s_w, n_samples_xy[0], dtype=np.int32)
        top_left_y = rng.integers(0, self._naip_h - s_h, n_samples_xy[1], dtype=np.int32)
        bottom_right_x = top_left_x + s_w
        bottom_right_y = top_left_y + s_h

        self._samples = self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        return self._samples

    def get_grid_samples(self):
        """
        Divide the NAIP image into grid samples and return the coordinates of
        the top-left and bottom-right of the grid samples.
        :return: np.ndarray of shape (n_samples, 4).
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y]. The block includes the bottom_right coordinates.
        """
        block_h = 512 if self.naip_h >= 512 else self.naip_h
        block_w = 512 if self.naip_w >= 512 else self.naip_w

        top_left_x = np.arange(0, self.naip_w - block_w, block_w, dtype=np.int32)
        top_left_y = np.arange(0, self.naip_h - block_h, block_h, dtype=np.int32)
        bottom_right_x = top_left_x + block_w
        bottom_right_y = top_left_y + block_h

        self._samples = self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        return self._samples

    def _make_diagonal_coordinates(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        top_left_xy = np.array(np.meshgrid(top_left_x, top_left_y)).T.reshape(-1, 2)
        bottom_right_xy = np.array(np.meshgrid(bottom_right_x, bottom_right_y)).T.reshape(-1, 2)
        diagonal_xy = np.concatenate((top_left_xy, bottom_right_xy), axis=1)
        return diagonal_xy

