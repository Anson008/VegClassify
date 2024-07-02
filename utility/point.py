class Point:
    def __init__(self, x: int, y: int):
        """
        Create an instance of the Point class
        :param x: int, the x coordinate of the point
        :param y: int, the y coordinate of the point
        """
        self._x = x
        self._y = y

    def __str__(self) -> str:
        return f"Point ({self.x}, {self.y}"

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, val: int):
        self._x = val

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, val: int):
        self._y = val
