import datetime
import numpy as np
import time

from simple_slurm import Slurm

# Configure Slurm object

slurm = Slurm(
    cpus_per_task=2,
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

file = 'a2c/d_simple_microgrid.py'

# Run list of experiments

experiments = [
    {
        "alr": 0.003288427767546812,
        "clr": 0.006038295776367509,
        "cnn": 256,
        "ann": 32,
    },
    {
        "alr": 0.0014994966516895213,
        "clr": 0.0028081176757678403,
        "cnn": 128,
        "ann": 128,
    }
]

for exp in experiments:

    exp_name = f"alr_{exp['alr']}_clr_{exp['clr']}_cnn_{exp['cnn']}_ann_{exp['ann']}"
    
    print(f"Starting with exp: {exp_name}")

    try:

        slurm.sbatch(f"python ./src/rl/{file}  -alr {exp['alr']} -clr {exp['clr']} -cnn {exp['cnn']} -ann {exp['ann']} -f {exp_name}")

    except:

        print("An exception occurred")
                
    time.sleep(np.random.randint(1, 10))

print(f"Finished!")