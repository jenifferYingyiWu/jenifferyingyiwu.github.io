from recsys.evaluation.baseclass import Evaluation
from recsys.evaluation import ROUND_FLOAT

# Decision-Based Metrics. Evaluating Top-N recommendations
class PrecisionRecallF1(Evaluation):
    def __init__(self):
        super(PrecisionRecallF1, self).__init__()

    def add_predicted_value(self, rating_pred): # Synonym of self.add_test
        self.add_test(rating_pred)

    def compute(self):
        super(PrecisionRecallF1, self).compute()
        hit_set = list(set(self._ground_truth) & set(self._test))
        precision = len(hit_set) / float(len(self._test))   # TP/float(TP+FP)
        recall = len(hit_set) / float(len(self._ground_truth))  # TP/float(TP+FN)
        if precision == 0.0 and recall == 0.0:
            return (0.0, 0.0, 0.0)
        f1 = 2 * ((precision * recall)/(precision + recall))
        return (round(precision, ROUND_FLOAT), round(recall, ROUND_FLOAT), round(f1, ROUND_FLOAT))