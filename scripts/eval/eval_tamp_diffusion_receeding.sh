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

function eval_tamp_diffusion {
    args=""
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    if [ ${#DIFFUSION_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --diffusion-checkpoints ${DIFFUSION_CHECKPOINTS[@]}"
    fi
    args="${args} --seed ${SEED}"
    # args="${args} --pddl-domain ${PDDL_DOMAIN}"
    # args="${args} --pddl-problem ${PDDL_PROBLEM}"
    args="${args} --max-depth 4"
    args="${args} --timeout 10"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 10"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 5"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_tamp_diffusion_trans_receeding.py ${args}"
    run_cmd
}

function run_planners {
    for planner in "${PLANNERS[@]}"; do

        POLICY_CHECKPOINTS=()
        for policy_env in "${POLICY_ENVS[@]}"; do
            if [[ "${planner}" == daf_* ]]; then
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${planner}/${policy_env}/${CKPT}.pt")
            else
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${policy_env}/${CKPT}.pt")
            fi
        done

        DIFFUSION_CHECKPOINTS=()
        for policy_env in "${POLICY_ENVS[@]}"; do
            if [[ "${planner}" == daf_* ]]; then
                DIFFUSION_CHECKPOINTS+=("${SCOD_INPUT_PATH}/${planner}/${policy_env}/${CKPT}.pt")
            else
                DIFFUSION_CHECKPOINTS+=("diffusion_models/v6_trans/unnormalized_${policy_env}/")
            fi
        done

        eval_tamp_diffusion
    done
}

function visualize_tamp {
    args=""
    args="${args} --path ${PLANNER_OUTPUT_PATH}"
    args="${args} --methods ${PLANNERS[@]}"
    CMD="python scripts/visualize/visualize_planners.py ${args}"
    run_cmd
}

SEED=100

# Setup.
DEBUG=0
input_path="models"
output_path="plots"

# Evaluate planners.
PLANNERS=(
    # "ablation/policy_cem"
    # "ablation/scod_policy_cem"
    # "ablation/policy_shooting"
    # # "daf_random_shooting"
    # "ablation/random_cem"
    # "ablation/random_shooting"
    # "greedy"
    "diffusion"
)

# Experiments.

# Pybullet.
exp_name="20221024/decoupled_state"
PLANNER_CONFIG_PATH="configs/pybullet/planners"
ENVS=(
    # "hook_reach/tamp0"
    # "hook_reach/tamp1"
    # "hook_reach/task0"
    # "hook_reach/task1"
    "hook_reach/task2"
    # "constrained_packing/tamp0"
    # "constrained_packing/task0"
    # "constrained_packing/task1"
    # "constrained_packing/task2"
    # "rearrangement_push/task0"
    # "rearrangement_push/task1"
    # "rearrangement_push/task2"
    # "rearrangement_push/task3"
)
POLICY_ENVS=("pick" "place" "pull" "push")
CKPT="ckpt_model_200000"
ENV_KWARGS="--closed-loop 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"
SCOD_INPUT_PATH="${input_path}/${exp_name}"
DYNAMICS_INPUT_PATH="${input_path}/${exp_name}"
# env1="constrained_packing/task2"
# env1="rearrangement_push/task3"
# env1="hook_reach/tamp0"
for env in "${ENVS[@]}"; do
    ENV_CONFIG="configs/pybullet/envs/official/domains/${env}.yaml"
    # PDDL_DOMAIN="configs/pybullet/envs/official/domains/${env}_domain.pddl"
    # PDDL_PROBLEM="configs/pybullet/envs/official/domains/${env}_problem.pddl"
    PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/tamp_experiment/${env}"
    run_planners
done

# Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [ $DEBUG -ne 0 ]; then
    exit
fi

PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/tamp_experiment"
# visualize_tamp
