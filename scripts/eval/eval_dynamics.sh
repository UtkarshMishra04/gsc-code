#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/train/train_gcp.sh "${CMD}"
    else
        ${CMD}
    fi
}

function train_dynamics {
    args=""
    args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10000"
    else
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}"
    fi

    CMD="python scripts/eval/eval_dynamics.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"

# Experiments.


# Pybox2d.

# exp_name="20220806/pybox2d"
# TRAINER_CONFIG="configs/pybox2d/trainers/dynamics.yaml"
# DYNAMICS_CONFIG="configs/pybox2d/dynamics/shared.yaml"
# policy_envs=("placeright" "pushleft")
# checkpoints=(
#     "final_model"
#     # "best_model"
#     # "ckpt_model_50000"
#     # "ckpt_model_100000"
# )

# Pybullet.

exp_name="20221024/decoupled_state"
DYNAMICS_CHECKPOINT="${exp_name}/ckpt_model_200000/dynamics"
TRAINER_CONFIG="${DYNAMICS_CHECKPOINT}/trainer_config.yaml"
DYNAMICS_CONFIG="${DYNAMICS_CHECKPOINT}/dynamics_config.yaml"
policy_envs=("pick" "place" "pull" "push")
checkpoints=(
    "ckpt_model_200000"
)
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

for ckpt in "${checkpoints[@]}"; do
    POLICY_CHECKPOINTS=()
    for policy_env in "${policy_envs[@]}"; do
        POLICY_CHECKPOINTS+=("${output_path}/${exp_name}/${policy_env}/${ckpt}.pt")
    done

    DYNAMICS_CHECKPOINT="${output_path}/${exp_name}/ckpt_model_200000/dynamics/final_model.pt"

    DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}/${ckpt}"
    train_dynamics
done
