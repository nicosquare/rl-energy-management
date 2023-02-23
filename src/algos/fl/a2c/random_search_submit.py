import datetime
import random
import numpy as np
import sys, time

from itertools import product
from simple_slurm import Slurm

# Configure Slurm object

slurm = Slurm(
    cpus_per_task=4,
    mem='60G',
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

nns = [32, 128, 256]
alrs = [0.0014994966516895213,0.003288427767546812]#np.random.uniform(high=1e-3, low=1e-4, size=n_params)
clrs = [0.0028081176757678403,0.006038295776367509]#np.random.uniform(high=1e-3, low=1e-4, size=n_params)
anns = random.sample(nns, n_params)
cnns = random.sample(nns, n_params)
sync_steps = [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

for sync_step, alr, clr, cnn, ann in product(sync_steps, alrs, clrs, cnns, anns):

    exp_name = f"{base_exp_name}_sync_step_{sync_step}_alr_{alr}_clr_{clr}_cnn_{cnn}_ann_{ann}"
    
    print(f"Starting with exp: {exp_name}")

    try:

        slurm.sbatch(f"python ./src/algos/fl/{file} -ss {sync_step} -alr {alr} -clr {clr} -cnn {cnn} -ann {ann} -f {exp_name}")

    except:

        print("An exception occurred")
                
    time.sleep(np.random.randint(1, 10))

print(f"Finished!")