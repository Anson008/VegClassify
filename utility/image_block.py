import numpy as np


class ImageBlock:
    def __init__(self, diagonal_xy: np.ndarray):
        """
        Create an instance of ImageBlock
        :param diagonal_xy: numpy array of shape (4, ) representing (tx, ty, bx, by),
        which are top-left and bottom-right coordinates in the original image from which
        the image block is sliced.
        """
        self._top_left_x = diagonal_xy[0]
        self._top_left_y = diagonal_xy[1]
        self._bottom_right_x = diagonal_xy[2]
        self._bottom_right_y = diagonal_xy[3]

    @property
    def top_left_x(self) -> int:
        return self._top_left_x.item()

    @property
    def top_left_y(self) -> int:
        return self._top_left_y.item()

    @property
    def bottom_right_x(self) -> int:
        return self._bottom_right_x.item()

    @property
    def bottom_right_y(self) -> int:
        return self._bottom_right_y.item()

    def get_all_coordinates(self) -> tuple[int, int, int, int]:
        return self.top_left_x, self.top_left_y, self.bottom_right_x, self.bottom_right_y

    def get_block_size(self) -> tuple[int, int]:
        """
        Get the size of an image block.
        :return: tuple, (height, width)
        """
        return self.bottom_right_y - self.top_left_y + 1, self.bottom_right_x - self.top_left_x + 1

    def get_relative_center(self) -> tuple[int, int]:
        """
        Get the center (x, y) coordinates of the image block.
        :return: tuple of int, (x, y).
        """
        x = (self.bottom_right_x - self.top_left_x) // 2
        y = (self.bottom_right_y - self.top_left_y) // 2
        return x, y

    def get_absolute_center(self) -> tuple[int, int]:
        """
        Get the center (x, y) coordinates of the image block in the original image (from which the block is cut) space.
        :return: tuple of int, (x, y).
        """
        x, y = self.get_relative_center()
        return x + self.top_left_x, y + self.top_left_y


if __name__ == "__main__":
    # diagonal_xy = np.array([[1, 2, 2, 1], [2, 3, 3, 2], [3, 4, 4, 3], [4, 5, 5, 5]])
    # for x in diagonal_xy:
    #     print(x.shape)

    a2 = np.array([1, 2, 3, 4])
    block = ImageBlock(a2)
    print(type(block.top_left_y), block.top_left_y)
