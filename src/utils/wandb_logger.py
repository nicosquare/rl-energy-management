import os
import wandb
import numpy as np

from dotenv import load_dotenv

class WandbLogger:

    def __init__(self, tags):
    
        # Load dotenv file

        current_path = os.path.abspath(os.path.dirname(__file__))
        load_dotenv(dotenv_path=os.path.join(current_path, "../../.env"))

        self.run = None
        self.project_name = str(os.environ.get("WANDB_PROJECT_NAME"))
        self.entity_name = str(os.environ.get("WANDB_ENTITY"))
        self.tags = tags
        self.enabled = True
        
    def init(self, config: dict=None, resume: str=None, run_id: int=None):

        if self.enabled:

            # Initialize Wandb for logging purposes

            wandb.login(key=str(os.environ.get("WANDB_KEY")))
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity_name,
                config=config,
                tags=self.tags,
                resume=resume,
                id=run_id
            )

    def disable_logging(self, disable):
        self.enabled = not disable

    def log_dict(self, dict, step: str = None):
        if self.enabled:
            wandb.log(dict, step=step)
        else:
            
            for key, value in dict.items():
                print(f"{key}: {value} \n", end="", flush=True)

    def log_multiline(self, name, xs, ys, keys, title, x_name):
        
        if self.enabled:

            wandb.log({
                f'{name}' : wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=keys,
                    title=title,
                    xname=x_name,
                )
            })

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

    def save_model(self):
        if self.enabled:

            wandb.save('model.pt')

    def load_model(self):
        if self.enabled:
            return wandb.restore('model.pt', run_path=f'{self.entity_name}/{self.project_name}/{self.run.id}')

    def finish(self):
        wandb.finish()