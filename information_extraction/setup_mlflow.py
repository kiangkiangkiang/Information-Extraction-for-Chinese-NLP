import mlflow
import sys
from paddlenlp.utils.log import logger
from paddle import nn, cast
import pandas as pd
from paddlenlp.metrics import SpanEvaluator


class ML_Flow_Handler(object):
    def __init__(
        self,
        exp_name="Information Extraction Task",
        run_tags={
            "user": "lab_luka",
            "project": "Verdict",
        },
        run_name="my run",
        description="Information Extraction in mlflow testing",
    ) -> None:
        self.ENV = sys.platform
        self.client = mlflow.client.MlflowClient()
        self.exp_id = None
        self.ENVIRONMENT_VARIABLE_SETUP_PATH = ""
        self.run_name = run_name
        self.run_tags = run_tags
        self.description = description
        self.exp_name = exp_name
        self.__setup_envir()
        self.__generate_exp_id()

    def __setup_envir(self):
        if self.ENV == "darwin":
            self.ENVIRONMENT_VARIABLE_SETUP_PATH = "/Users/cfh00892302/Desktop/myWorkspace/AWS_private/"
        elif self.ENV == "linux":
            # TODO ubuntu
            self.ENVIRONMENT_VARIABLE_SETUP_PATH = ""
        else:
            raise ValueError(f"OS of {sys.platform} is not be support.")

        # setup by environment
        sys.path.append(self.ENVIRONMENT_VARIABLE_SETUP_PATH)
        """
        try:
            from setup_mlflow_envir import setup_env

            self.setup_env = setup_env
        except Exception as e:
            raise ValueError(f"Cannot setup mlflow. {e.__class__.__name__}: {e}.")
        """

    def __generate_exp_id(self) -> str:
        if self.exp_id:
            return
        try:
            self.exp_id = self.client.create_experiment(self.exp_name)
        except mlflow.exceptions.MlflowException:
            # experiment name already exist!
            exp = self.client.get_experiment_by_name(self.exp_name)
            self.exp_id = exp.experiment_id

        logger.debug(f"exp_id={self.exp_id}")

    def mlflow_train(self, trainer, checkpoint, log_parms_dict):
        with mlflow.start_run(
            run_name=self.run_name,
            experiment_id=self.exp_id,
            tags=self.run_tags,
            description=self.description,
        ):
            mlflow.log_params(log_parms_dict)
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            return train_result

    def loss_func(self, outputs, labels) -> float:
        loss_func = nn.BCELoss()
        start_ids, end_ids = labels
        start_prob, end_prob = outputs
        start_ids = cast(start_ids, "float32")
        end_ids = cast(end_ids, "float32")
        loss_start = loss_func(start_prob, start_ids)
        loss_end = loss_func(end_prob, end_ids)
        loss = (loss_start + loss_end) / 2.0
        logger.debug("in mlflow loss")
        mlflow.log_metric(key="train loss", value=loss)
        return loss

    def SpanEvaluator_metrics(self, result):
        logger.debug("in mlflow metrics")
        metric = SpanEvaluator()

        def compute_metrics(predictions, label_ids, descriptions=""):

            start_prob, end_prob = predictions
            logger.debug(f"predictions shape: s:{start_prob.shape}, e:{end_prob.shape}")
            start_ids, end_ids = label_ids
            logger.debug(f"label_ids shape: s:{start_ids.shape}, e:{end_ids.shape}")
            metric.reset()
            num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
            metric.update(num_correct, num_infer, num_label)
            precision, recall, f1 = metric.accumulate()
            metric.reset()
            mlflow.log_metric(key=descriptions + "_precision", value=precision)
            mlflow.log_metric(key=descriptions + "_recall", value=recall)
            mlflow.log_metric(key=descriptions + "_f1", value=f1)
            return {descriptions + "precision": precision, descriptions + "recall": recall, descriptions + "f1": f1}

        if result.eval_by_group:
            ner_type = pd.unique(result.eval_group)
            num_type = len(ner_type)
            if num_type < 1:
                raise ValueError("Cannot compute metrics by eval_group when length of ner_type < 1.")
            else:
                metric_result = {}
                # compute by group
                for each_type in ner_type:
                    selected_group = result.eval_group == each_type
                    logger.debug(f"current ner_type={each_type}")
                    logger.debug(f"length of current ner_type={sum(selected_group)}")
                    metric_result.update(
                        compute_metrics(
                            predictions=tuple(result.predictions[i][selected_group, :] for i in (0, 1)),
                            label_ids=tuple(result.label_ids[i][selected_group, :] for i in (0, 1)),
                            descriptions=each_type + "_",
                        )
                    )
                # compute total
                Total_Precision, Total_Recall, Total_F1 = 0, 0, 0
                tmp_metrics = list(metric_result.values())
                runs = int(len(tmp_metrics) / num_type)
                for i in range(0, len(tmp_metrics), runs):

                    Total_Precision += tmp_metrics[i]
                    Total_Recall += tmp_metrics[i + 1]
                    Total_F1 += tmp_metrics[i + 2]
                metric_result.update(
                    {
                        "precision": Total_Precision / num_type,
                        "recall": Total_Recall / num_type,
                        "f1": Total_F1 / num_type,
                    }
                )
                return metric_result
        else:
            return compute_metrics(predictions=result.predictions, label_ids=result.label_ids)

    def setup_env(self):
        pass
