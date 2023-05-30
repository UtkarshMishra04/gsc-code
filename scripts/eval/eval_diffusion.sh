#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/eval/eval_planners_juno.sh "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/eval/eval_gcp.sh "${CMD}"
    else
        ${CMD}
    fi
}

function eval_diffusion {
    args=""
    args="${args} --checkpoint ${POLICY_CHECKPOINT}"
    args="${args} --diffusion-checkpoint ${DIFFUSION_CHECKPOINT}"
    # args="${args} --env-config ${ENV_CONFIG}"
    args="${args} --seed ${SEED}"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --num-episodes 1"
        args="${args} --verbose 1"
    else
        args="${args} --path plots/${EXP_NAME}"
        args="${args} --verbose 0"
        args="${args} --num-episodes ${NUM_EPISODES}"
    fi
    if [[ -n "${DEBUG_RESULTS}" ]]; then
        args="${args} --debug-results ${DEBUG_RESULTS}"
    fi
    if [[ -n "${ENV_CONFIG}" ]]; then
        args="${args} --env-config ${ENV_CONFIG}"
    fi
    CMD="python scripts/eval/eval_diffusion.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0
NUM_EPISODES=5

# Evaluate policies.

SEED=0

policy_envs=(
    "pick"
    # "place"
    # "pull"
    # "push"
)
experiments=(
    "20221024/decoupled_state" # "20220908/official"
)
ckpts=(
    # "best_model"
    # "final_model"
    # "ckpt_model_50000"
    # "ckpt_model_100000"
    # "ckpt_model_150000"
    "ckpt_model_200000"
)
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="${ENV_KWARGS} --gui 0"
fi

for exp_name in "${experiments[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        for policy_env in "${policy_envs[@]}"; do
            EXP_NAME="${exp_name}/${ckpt}"
            POLICY_CHECKPOINT="models/${exp_name}/${policy_env}/${ckpt}.pt"
            DIFFUSION_CHECKPOINT="diffusion_models/v3/unnormalized_${policy_env}/"
            ENV_CONFIG="configs/pybullet/envs/official/primitives/${policy_env}_eval.yaml"
            eval_diffusion
        done
    done
done
