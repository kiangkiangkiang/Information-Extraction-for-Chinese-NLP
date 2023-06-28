from paddle import nn, cast
from paddlenlp.metrics import SpanEvaluator

loss_function = nn.BCELoss()


def uie_loss_func(outputs, labels):
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    loss_start = loss_function(start_prob, start_ids)
    loss_end = loss_function(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


def compute_metrics(p):
    metric = SpanEvaluator()
    start_prob, end_prob = p.predictions
    start_ids, end_ids = p.label_ids
    metric.reset()
    num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    metric.reset()
    return {"precision": precision, "recall": recall, "f1": f1}
