import numpy as np


class ImageBlock:
    def __init__(self, diagonal_xy: np.ndarray):
        self._top_left_x = diagonal_xy[0]
        self._top_left_y = diagonal_xy[1]
        self._bottom_right_x = diagonal_xy[2]
        self._bottom_right_y = diagonal_xy[3]

    @property
    def top_left_x(self):
        return self._top_left_x

    @property
    def top_left_y(self):
        return self._top_left_y

    @property
    def bottom_right_x(self):
        return self._bottom_right_x

    @property
    def bottom_right_y(self):
        return self._bottom_right_y

    def get_all_coordinates(self):
        return self._top_left_x, self._top_left_y, self._bottom_right_x, self._bottom_right_y

    def get_block_size(self):
        """
        Get the size of an image block
        :return: tuple, (height, width)
        """
        return self._bottom_right_y - self._top_left_y + 1, self._bottom_right_x - self._top_left_x + 1

    def get_relative_center(self):
        x = (self._bottom_right_x - self._top_left_x) // 2
        y = (self._bottom_right_y - self._top_left_y) // 2
        return x, y

    def get_absolute_center(self):
        x, y = self.get_relative_center()
        return x + self._top_left_x, y + self._top_left_y
