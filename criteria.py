from abc import ABC, abstractmethod


class CriteriaBase(ABC):
    def __init__(self, param, target):
        self._param = param
        self._target = target

    @abstractmethod
    def produce_criteria(self):
        pass


class GreaterThanOrEqualToCriteria(CriteriaBase):

    def produce_criteria(self):
        return self._param >= self._target


class GreaterThanCriteria(CriteriaBase):
    def produce_criteria(self):
        return self._param > self._target


class LessThanOrEqualToCriteria(CriteriaBase):

    def produce_criteria(self):
        return self._param <= self._target


class LessThanCriteria(CriteriaBase):
    def produce_criteria(self):
        return self._param < self._target


class AndCriteria(CriteriaBase):
    def produce_criteria(self):
        return self._param and self._target


class OrCriteria(CriteriaBase):
    def produce_criteria(self):
        return self._param or self._target

