#!/bin/bash

source ~/.bashrc
source activate hi_diffuser

sub_conf='maze2d_lg_ev_prob_smoke'
sub_conf='maze2d_lg_ev_prob_smoke_2'
sub_conf='maze2d_lg_ev_prob_bt2way_nppp3'
sub_conf='maze2d_lg_ev_prob_bt2way_nppp3_rprange02'



{

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python scripts/eval_problems/generate_maze2d_eval_problems.py \
    --sub_conf ${sub_conf}


exit 0

}
