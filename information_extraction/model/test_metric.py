from paddle.metric import Metric
from paddlenlp.utils.tools import get_bool_ids_greater_than, get_span
from paddlenlp.utils.log import logger


class SpanEvaluator(Metric):
    """
    SpanEvaluator computes the precision, recall and F1-score for span detection.
    """

    def __init__(self, limit=0.5):
        super(SpanEvaluator, self).__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0
        self.limit = limit

    def compute(self, start_probs, end_probs, gold_start_ids, gold_end_ids):
        """
        Computes the precision, recall and F1-score for span detection.
        """
        pred_start_ids = get_bool_ids_greater_than(start_probs, self.limit)
        pred_end_ids = get_bool_ids_greater_than(end_probs, self.limit)
        gold_start_ids = get_bool_ids_greater_than(gold_start_ids.tolist(), self.limit)
        gold_end_ids = get_bool_ids_greater_than(gold_end_ids.tolist(), self.limit)
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0
        for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
            pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids
        ):
            [_correct, _infer, _label] = self.eval_span(
                predict_start_ids, predict_end_ids, label_start_ids, label_end_ids
            )
            num_correct_spans += _correct
            num_infer_spans += _infer
            num_label_spans += _label
            if _correct > 0 or _infer > 0:
                logger.debug(
                    f"(num_correct, num_infer, num_label): {(num_correct_spans, num_infer_spans, num_label_spans)}"
                )
        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def eval_span(self, predict_start_ids, predict_end_ids, label_start_ids, label_end_ids):
        """
        evaluate position extraction (start, end)
        return num_correct, num_infer, num_label
        input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
        output: (1, 2, 2)
        """
        pred_set = get_span(predict_start_ids, predict_end_ids)
        label_set = get_span(label_start_ids, label_end_ids)

        # Debug
        if len(pred_set) > 0 or len(label_set) > 0:
            if pred_set == label_set:
                logger.info(f"Correct in: predict: {pred_set} == true: {label_set}")
            else:
                logger.error(f"Error in: predict: {pred_set} != true: {label_set}")

        num_correct = len(pred_set & label_set)
        num_infer = len(pred_set)
        # For the case of overlapping in the same category,
        # length of label_start_ids and label_end_ids is not equal
        num_label = max(len(label_start_ids), len(label_end_ids))
        return (num_correct, num_infer, num_label)

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_spans / self.num_infer_spans) if self.num_infer_spans else 0.0
        recall = float(self.num_correct_spans / self.num_label_spans) if self.num_label_spans else 0.0
        f1_score = float(2 * precision * recall / (precision + recall)) if self.num_correct_spans else 0.0
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"