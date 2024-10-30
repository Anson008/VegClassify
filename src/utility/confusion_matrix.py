class ConfusionMatrix:
    def __init__(self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

    @property
    def tp(self):
        return self._tp

    @tp.setter
    def tp(self, value):
        self._tp = value

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, value):
        self._fp = value

    @property
    def tn(self):
        return self._tn

    @tn.setter
    def tn(self, value):
        self._tn = value

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        self._fn = value

    def get_kappa(self):
        try:
            numerator = self._tp * self._tn - self._fp * self._fn
            denominator = ((self._tp + self._fp) * (self._fp + self._tn) +
                       (self._tp + self._fn) * (self._fn + self._tn))
            return 2.0 * numerator / denominator
        except ZeroDivisionError as err:
            print(f"{err}: Failed to calculate kappa")
            return -1

    def get_accuracy(self):
        try:
            return 1.0 * (self._tp + self._tn) / (self._tp + self._fp + self._tn + self._fn)
        except ZeroDivisionError as err:
            print(f"{err}: Failed to calculate accuracy")
            return -1

    def get_confusion_matrix(self):
        return {"tp": self._tp,
                "fp": self._fp,
                "tn": self._tn,
                "fn": self._fn,
                "accuracy": self.get_accuracy(),
                "kappa": self.get_kappa()}

