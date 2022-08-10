import os
import wandb

from dotenv import load_dotenv

class WandbLogger:

    def __init__(self, project_name):
    
        # Load dotenv file

        current_path = os.path.abspath(os.path.dirname(__file__))
        load_dotenv(dotenv_path=os.path.join(current_path, "../../.env"))

        self.project_name = project_name
        self.entity_name = str(os.environ.get("WANDB_ENTITY"))
        self.enabled = True
        
    def init(self, config):
        if self.enabled:

            # Initialize Wandb for logging purposes

            wandb.login(key=str(os.environ.get("WANDB_KEY")))
            wandb.init(project=self.project_name, entity=self.entity_name, config=config)

    def disable_logging(self, disable):
        self.enabled = not disable

    def log_dict(self, dict, step: str = None):
        if self.enabled:
            wandb.log(dict, step=step)

    def log_scalar(self, name, value, step):
        if self.enabled:
            wandb.log({name: value}, step=step)

    def log_histogram(self, name, values, step):
        if self.enabled:
            wandb.log({name: values}, step=step)

    def log_figure(self, name, figure, step):
        if self.enabled:
            wandb.log({name: figure}, step=step)

    def watch_model(self, models):
        if self.enabled:
            wandb.watch(models=models)

    def finish(self):
        wandb.finish()