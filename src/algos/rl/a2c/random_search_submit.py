import datetime
import random
import numpy as np
import sys, time

from itertools import product
from simple_slurm import Slurm

# Configure Slurm object

slurm = Slurm(
    cpus_per_task=4,
    mem='40G',
    qos='cpu-4',
    partition='cpu',
    job_name='BCTE',
    output=f'./logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    error=f'./logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.err',
    time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
)

# slurm = Slurm(
#     cpus_per_task=32,
#     mem='20G',
#     gres='gpu:1',
#     qos='gpu-8',
#     partition='gpu',
#     job_name='BCTE',
#     output=f'./logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
#     error=f'./logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.err',
#     time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
# )

file = 'a2c/d_simple_mg.py'
base_exp_name = ''

assert(base_exp_name == '', "Experiment name cannot be empty!")

# Perform random exploration of hyperparameters

n_params = 3

nns = [32, 64, 128, 256, 512]
alrs = np.random.uniform(high=1e-3, low=1e-4, size=n_params)
clrs = np.random.uniform(high=1e-3, low=1e-4, size=n_params)
anns = random.sample(nns, n_params)
cnns = random.sample(nns, n_params)

for alr, clr, cnn, ann in product(alrs, clrs, cnns, anns):

    exp_name = f"{base_exp_name}_alr_{alr}_clr_{clr}_cnn_{cnn}_ann_{ann}"
    
    print(f"Starting with exp: {exp_name}")

    try:

        slurm.sbatch(f"python ./src/algos/rl/{file}  -alr {alr} -clr {clr} -cnn {cnn} -ann {ann} -f {exp_name}")

    except:

        print("An exception occurred")
                
    time.sleep(np.random.randint(1, 10))

print(f"Finished!")