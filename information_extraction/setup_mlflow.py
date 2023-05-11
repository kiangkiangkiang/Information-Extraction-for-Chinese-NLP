import mlflow
import sys
from paddlenlp.utils.log import logger
from paddle import nn, cast


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
        try:
            from setup_mlflow_envir import setup_env

            self.setup_env = setup_env
        except Exception as e:
            raise ValueError(f"Cannot setup mlflow. {e.__class__.__name__}: {e}.")

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
        mlflow.log_metric(key="test 87 log", value=878787)
        return loss

    def setup_env(self):
        pass
