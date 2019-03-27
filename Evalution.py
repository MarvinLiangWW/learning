import numpy as np
import math


class Evalution(object):
    def __init__(self):
        pass

    def mrr(self, y_true: np.array, y_pred: np.array):
        coupled_pair = self.sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair):
            # if label > self._threshold:
            if label > 0:
                return 1. / (idx + 1)
        return 0.

    def map(self, y_true: np.array, y_pred: np.array):
        # threshold: The threshold of relevance degree.
        result = 0.
        pos = 0
        coupled_pair = self.sort_and_couple(y_true, y_pred)
        for idx, (label, score) in enumerate(coupled_pair):
            # if label > self._threshold:
            if label > 0:
                pos += 1.
                result += pos / (idx + 1.)
        if pos == 0:
            return 0.
        else:
            return result / pos

    @staticmethod
    def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
        """Zip the `labels` with `scores` into a single list."""
        couple = list(zip(labels, scores))
        return np.array(sorted(couple, key=lambda x: x[1], reverse=True))

    def dcg_at_k(self, y_true: np.array, y_pred: np.array, k):
        if k <= 0:
            return 0.
        coupled_pair = self.sort_and_couple(y_true, y_pred)
        result = 0.
        for i, (label, score) in enumerate(coupled_pair):
            if i >= k:
                break
            # if label >self._threshold:
            if label > 0.:
                result += (math.pow(2., label) - 1.) / math.log(2. + i)
        return result

    def ndcg_at_k(self, y_true: np.array, y_pred: np.array, k):
        dcg_max = self.dcg_at_k(y_true, y_true, k)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(y_true, y_pred, k) / dcg_max
