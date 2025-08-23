import wandb
import avion.utils.distributed as dist_utils

class WandBManager:
    def __init__(self, project_name, entity=None, config=None):
        """
        Initialize the WandBManager.

        Args:
            project_name (str): The name of the project in WandB.
            entity (str, optional): The entity or team name in WandB. Defaults to None.
            config (dict, optional): Configuration dictionary to log in WandB. Defaults to None.
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config
        self.run = None

    def start_run(self, run_name=None):
        """
        Start a new WandB run.

        Args:
            run_name (str, optional): The name of the run. Defaults to None.
        """
        if not dist_utils.is_main_process():
            return
        
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            name=run_name,
            resume="allow"
        )

    def log_metrics(self, metrics, step=None):
        """
        Log metrics to the current WandB run.

        Args:
            metrics (dict): A dictionary of metrics to log.
            step (int, optional): The step number for the metrics. Defaults to None.
        """
        if not dist_utils.is_main_process():
            return
        if self.run is not None:
            wandb.log(metrics, step=step)
        else:
            raise RuntimeError("WandB run has not been started. Call start_run() first.")

    def finish_run(self):
        """
        Finish the current WandB run.
        """
        if not dist_utils.is_main_process():
            return
        if self.run is not None:
            self.run.finish()
            self.run = None
        else:
            raise RuntimeError("WandB run has not been started or is already finished.")

    def watch_model(self, model, log="gradients", log_freq=100):
        """
        Watch a model to log gradients and parameters.

        Args:
            model: The model to watch.
            log (str, optional): What to log ("gradients", "parameters", or "all"). Defaults to "gradients".
            log_freq (int, optional): Frequency of logging. Defaults to 100.
        """
        if not dist_utils.is_main_process():
            return
        if self.run is not None:
            self.run.watch(model, log=log, log_freq=log_freq)
        else:
            raise RuntimeError("WandB run has not been started. Call start_run() first.")
