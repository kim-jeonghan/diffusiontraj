#!/bin/bash

#SBATCH --job-name=script-ev-hi
#SBATCH --output=trash/slurm/plan_StglSml/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude="clippy,voltron, claptrap"
###SBATCH --partition="rl2-lab"

##SBATCH --gres=gpu:a40:1
#SBATCH --gres=gpu:rtx_6000:1
##
#SBATCH --qos="debug"
#SBATCH --time=5:00:00
##
###SBATCH --qos="debug"
###SBATCH --time=48:00:00

source ~/.bashrc
source activate hi_diffuser
cd /coc/flash7/yluo470/robot2024/hi_src/comp_diffuser/
echo $(hostname)

#### Jan 18
#### Ben
conda activate hi_diffuser_ben
config=""
config="configs/experiment/maze2d/maze2d_umaze_baseline_config.py" ## hzn=136

# config="configs/experiment/maze2d/maze2d_medium_baseline_config.py"
# config="configs/experiment/maze2d/maze2d_large_baseline_config.py"


{

# PYTHONBREAKPOINT=0 \ ## -B
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python scripts/plan_maze_baseline.py \
    --config $config \
    --plan_n_ep $2 \
    --pl_seeds ${3:--1} \


exit 0

}
