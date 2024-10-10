from naip.sample_method import SampleMethod


class NaipSampler:
    def __init__(self, sample_method: SampleMethod):
        """
        Create an instance of NaipSampler.
        :param naip_h: int, height of the input NAIP imagery
        :param naip_w: int, width of the input NAIP imagery
        """
        self._sample_method = sample_method

    @property
    def sample_method(self):
        return self._sample_method

    @sample_method.setter
    def sample_method(self, sample_method: SampleMethod):
        self._sample_method = sample_method

    def get_sample_coordinates(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        return self._sample_method.sample(naip_size, sample_shape)
