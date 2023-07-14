import math


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

