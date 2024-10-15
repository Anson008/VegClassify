from naip.sample_method import SampleMethod


class NaipSampler:
    def __init__(self, sample_method: SampleMethod):
        """
        Create an instance of NaipSampler.
        :param sample_method: SampleMethod, specify the sample method that the sampler uses.
        """
        self._sample_method = sample_method

    @property
    def sample_method(self):
        """
        Get current SampleMethod object.
        :return: SampleMethod, current SampleMethod for the NaipSampler.
        """
        return self._sample_method

    @sample_method.setter
    def sample_method(self, sample_method: SampleMethod):
        """
        Set current SampleMethod.
        :param sample_method: SampleMethod, the SampleMethod to set.
        :return: None.
        """
        self._sample_method = sample_method

    def get_sample_coordinates(self, naip_size: tuple[int, int], sample_shape: tuple[int, int]):
        """
        Get the top-left and bottom-right coordinates of the samples.
        :param naip_size: tuple of int (height, width), size of the NAIP imagery to sample from.
        :param sample_shape: tuple of int (height, width), size of an individual sample.
        :return: np.ndarray of shape (n_samples, 4), top-left and bottom-right coordinates of the samples.
         Each row represents [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
        """
        return self._sample_method.sample(naip_size, sample_shape)
