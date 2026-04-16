#!/bin/bash

#SBATCH --job-name=script-hi-dd-bl
#SBATCH --output=trash/slurm/train_dd_maze_bline/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude="clippy,voltron"
###SBATCH --partition="rl2-lab"

#SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:a40:1
###SBATCH --gres=gpu:rtx_6000:1
##
##SBATCH --qos="long"
##SBATCH --time=72:00:00
##
#SBATCH --qos="debug"
#SBATCH --time=48:00:00

source ~/.bashrc
source activate hi_diffuser
cd /coc/flash7/yluo470/robot2024/hi_src/comp_diffuser/

## Jan 17
config="configs/experiment/maze2d/maze2d_umaze_baseline_smoke_config.py"
# config="configs/experiment/maze2d/maze2d_umaze_baseline_config.py"

# config="configs/experiment/maze2d/maze2d_medium_baseline_config.py"
# config="configs/experiment/maze2d/maze2d_large_baseline_config.py"

{

# PYTHONBREAKPOINT=0 \ ## -B
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python scripts/train_maze_baseline.py --config $config \


exit 0

}
