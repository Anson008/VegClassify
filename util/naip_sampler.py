import numpy as np


class NaipSampler:
    def __init__(self, naip_h, naip_w):
        self._naip_h = naip_h
        self._naip_w = naip_w

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

    def get_random_naip_imagery_samples(self, n_samples_xy=(2, 2), seed=None):
        s_w = min(int(self._naip_w / n_samples_xy[0]), 256)
        s_h = min(int(self._naip_h / n_samples_xy[1]), 512)

        rng = np.random.default_rng(seed)
        top_left_x = rng.integers(0, self._naip_w - s_w, n_samples_xy[0], dtype=np.int32)
        top_left_y = rng.integers(0, self._naip_h - s_h, n_samples_xy[1], dtype=np.int32)
        bottom_right_x = top_left_x + s_w
        bottom_right_y = top_left_y + s_h

        return self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    def get_grid_center(self):
        block_h = 512 if self.naip_h >= 512 else self.naip_h
        block_w = 256 if self.naip_w >= 256 else self.naip_w

        bottom_right_x = np.arange(block_w, self.naip_w, block_w, dtype=np.int32)
        bottom_right_y = np.arange(block_h, self.naip_h, block_h, dtype=np.int32)
        top_left_x = bottom_right_x - block_w
        top_left_y = bottom_right_y - block_h

        return self._make_diagonal_coordinates(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    def _make_diagonal_coordinates(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        top_left_xy = np.array(np.meshgrid(top_left_x, top_left_y)).T.reshape(-1, 2)
        bottom_right_xy = np.array(np.meshgrid(bottom_right_x, bottom_right_y)).T.reshape(-1, 2)
        diagonal_xy = np.concatenate((top_left_xy, bottom_right_xy), axis=1)
        return diagonal_xy