from abc import ABC, abstractmethod


class FilterBase(ABC):
    def __init__(self, param, target):
        self._param = param
        self._target = target

    @abstractmethod
    def produce_filter(self):
        pass


class GreaterThanOrEqualToFilter(FilterBase):

    def produce_filter(self):
        return self._param >= self._target


class GreaterThanFilter(FilterBase):
    def produce_filter(self):
        return self._param > self._target


class LessThanOrEqualToFilter(FilterBase):

    def produce_filter(self):
        return self._param <= self._target


class LessThanFilter(FilterBase):
    def produce_filter(self):
        return self._param < self._target
