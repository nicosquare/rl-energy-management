import argparse
import wandb

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from src.environments.mg_source_selection import MGSourceSelection

"""
    Main method definition

"""

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--LearningRate", help="Learning rate of the NN")
parser.add_argument("-m", "--MiniBatch", help="Mini batch size")

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    # Parse the parameters

    LEARNING_RATE = float(args.LearningRate) if args.LearningRate else 1e-4
    MINIBATCH_SIZE = int(args.MiniBatch) if args.MiniBatch else 16

    # Configure environment for SB

    env = Monitor(MGSourceSelection())

    run = wandb.init(
        project='sb_dqn_mg',
        entity="madog",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    model_save_path = f"./wandb/sb_dqn_sb/dqn_mg_/{run.id}"

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        learning_starts=100,
        batch_size=MINIBATCH_SIZE,
        device='cpu'
    )

    model.learn(
        total_timesteps=24*365,
        n_eval_episodes=10,
        log_interval=4,
        callback=WandbCallback(
            model_save_freq=100,
            verbose=1,
            gradient_save_freq=10,
            model_save_path=model_save_path
        )
    )

    run.finish()

