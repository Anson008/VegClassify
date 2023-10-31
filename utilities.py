import math
import ctypes


class UnitConverterFactory:
    @staticmethod
    def create_converter(converter_type):
        try:
            if converter_type == "DegreeToRadian":
                return DegreeToRadian()
            elif converter_type == "RadianToDegree":
                return RadianToDegree()
            raise AssertionError("Converter type is not valid")
        except AssertionError as e:
            print(e)


class DegreeToRadian:
    @staticmethod
    def get_factor():
        return 2 * math.pi / 360.0


class RadianToDegree:
    @staticmethod
    def get_factor():
        return 360.0 / (2 * math.pi)


def get_screen_resolution():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    return width, height


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __str__(self):
        return f"Point ({self.x}, {self.y}"

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

