#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import symbolic
import tqdm
import json
from PIL import Image
import pickle
from temporal_policies import dynamics, agents, envs, planners, agents
from temporal_policies.envs.pybullet.table import primitives as table_primitives
from temporal_policies.utils import recording, timing, random, tensors

from temporal_policies.diff_models.unet_transformer import ScoreNet, ScoreNetState
from temporal_policies.diff_models.classifier_transformer import ScoreModelMLP, TransitionModel
from temporal_policies.mixed_diffusion.cond_diffusion1D import Diffusion

@tensors.numpy_wrap
def query_policy_actor(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    # print("policy.encoder:", policy.encoder)
    # print("policy.encoder.encode(observation.to(policy.device), policy_args):", policy.encoder.encode(observation.to(policy.device), policy_args))
    return policy.actor.predict(
        policy.encoder.encode(observation.to(policy.device), policy_args)
    )

@tensors.numpy_wrap
def query_observation_vector(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.encoder.encode(observation.to(policy.device), policy_args)


def observation_str(env: envs.Env, observation: np.ndarray) -> str:
    """Converts observations to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        return str(env.object_states())

    return str(observation)


def action_str(env: envs.Env, action: np.ndarray) -> str:
    """Converts actions to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        primitive = env.get_primitive()
        assert isinstance(primitive, envs.pybullet.table.primitives.Primitive)
        return str(primitive.Action(action))

    return str(action)

def transform_forward(
    observation: torch.Tensor,
    indices: np.ndarray,
) -> torch.Tensor:

    if len(observation.shape) == 1:
        curr_size = observation.shape[0]
        return observation.reshape(8, 12)[indices].reshape(curr_size)

    curr_size = observation.shape[1]
    
    return observation.reshape(-1, 8, 12)[:, indices].reshape(-1, curr_size)

def transform_backward(
    observation: torch.Tensor,
    indices: np.ndarray,
) -> torch.Tensor:

    if len(observation.shape) == 1:
        curr_size = observation.shape[0]
        return observation.reshape(8, 12)[indices].reshape(curr_size)

    curr_size = observation.shape[1]

    return observation.reshape(-1, 8, 12)[:, indices].reshape(-1, curr_size)

def forward_diffusion(
    diffusion_model: Diffusion,
    observation: torch.Tensor,
    observation_indices: np.ndarray,
    reset_observation_indices: np.ndarray,
    device: torch.device = "auto",
    num_samples: int = 5,
    num_objects: int = 7,
    end_index: int = 36,
    num_steps: int = 256,
    state_dim: int = 96,
    action_dim: int = 4,
):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    diffusion_model.to(device)

    all_samples = []

    if observation is not None:
        obs0 = np.array(observation)*2
        x0 = torch.Tensor(obs0).to(device)
    else:
        x0 = None

    reverse_observation_indices = np.zeros_like(observation_indices)

    for j in range(len(observation_indices)):
        reverse_observation_indices[observation_indices[j]] = j

    reverse_reset_observation_indices = np.zeros_like(reset_observation_indices)

    for j in range(len(reset_observation_indices)):
        reverse_reset_observation_indices[reset_observation_indices[j]] = j

    obs_ind = torch.Tensor(observation_indices).to(device).unsqueeze(0).repeat(num_samples, 1)
    
    if x0 is not None:
        mod_x0 = transform_backward(x0, reverse_reset_observation_indices)
    else:
        mod_x0 = None
    # mod_x0 = transform_forward(x0, observation_indices)

    # all_samples.append(observation.clone().unsqueeze(0).cpu().numpy())

    sample_dim = state_dim + action_dim + state_dim

    xt = torch.zeros(num_samples, sample_dim).to(device)

    sde, ones = diffusion_model.configure_sdes(num_steps=num_steps, x_T=xt, num_samples=num_samples)

    all_samples.append(xt.clone().unsqueeze(0).cpu().numpy())

    for t in range(num_steps):

        sample = xt.clone()
        sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices)
        sample[:, -state_dim:] = transform_forward(sample[:, -state_dim:], observation_indices)

        epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, obs_ind)
        
        pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
        
        pred_x0[:, -state_dim+36:] = pred_x0[:, 36:state_dim]

        epsilon = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

        epsilon[:, :state_dim] = transform_backward(epsilon[:, :state_dim], reverse_observation_indices)
        epsilon[:, -state_dim:] = transform_backward(epsilon[:, -state_dim:], reverse_observation_indices)

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)

        if mod_x0 is not None:
            pred_x0[:, 12:end_index] = mod_x0[12:end_index]
            pred_x0[:, 12*num_objects:state_dim] = mod_x0[12*num_objects:]
            # pred_x0[:, :state_dim] = mod_x0[:state_dim]
            pred_x0[:, -state_dim+12:-state_dim+end_index] = mod_x0[12:end_index]
            pred_x0[:, -state_dim+12*num_objects:] = mod_x0[12*num_objects:]

        with torch.no_grad():

            pred_x0 = torch.clip(pred_x0, -1, 1)
            new_epsilon = torch.randn_like(epsilon)
            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

        all_samples.append(xt.clone().unsqueeze(0).cpu().numpy())
    
    all_samples = np.concatenate(all_samples, axis=0)

    all_mod_samples = []

    for i in range(num_samples):
        all_samples1 = all_samples[:, i, :]

        # all_samples1[:, :state_dim] = transform_backward(all_samples1[:, :state_dim], reverse_observation_indices)*0.5
        # all_samples1[:, -state_dim:] = transform_backward(all_samples1[:, -state_dim:], reverse_observation_indices)*0.5

        all_samples1[:, :state_dim] = all_samples1[:, :state_dim]*0.5
        all_samples1[:, -state_dim:] = all_samples1[:, -state_dim:]*0.5

        all_mod_samples.append(all_samples1)

    return all_mod_samples

def get_action_from_multi_diffusion(
    policies: Sequence[agents.RLAgent],
    diffusion_models: Sequence[Diffusion],
    transition_models: Sequence[TransitionModel],
    classifiers: Sequence[ScoreModelMLP],
    obs0: torch.Tensor,
    action_skeleton: Sequence[envs.Primitive],
    use_transition_model: bool = True,
    num_samples: int = 40,
    num_objects: int = 4,
    end_index: int = 24,
    state_dim: int = 96,
    action_dim: int = 4,
    gamma: Sequence[float] = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
    device: torch.device = "auto"
) -> np.ndarray:

    start_time = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    for policy, diffusion_model, transition_model, classifier in zip(policies, diffusion_models, transition_models, classifiers):
        policy.to(device)
        diffusion_model.to(device)
        transition_model.to(device)
        classifier.to(device)

    num_steps = 256
    sample_dim = 0
    for i in range(len(policies)):
        sample_dim += state_dim + action_dim
    sample_dim += state_dim

    indices_dms = []
    indices_sdms = []

    for i in range(len(policies)):
        indices_dms.append((i*(state_dim+action_dim), (i+1)*(state_dim+action_dim)+state_dim))
        indices_sdms.append((i*(state_dim+action_dim), (i)*(state_dim+action_dim)+state_dim))

    xt = torch.zeros(num_samples, sample_dim).to(device)

    all_observation_indices = []
    all_reverse_observation_indices = []

    print("action_skeleton:", len(action_skeleton))
    print("policies:", len(policies))

    for i in range(len(policies)):
        observation_indices = np.array(action_skeleton[i].get_policy_args()["observation_indices"])
        reverse_observation_indices = np.zeros_like(observation_indices)

        for j in range(len(observation_indices)):
            reverse_observation_indices[observation_indices[j]] = j

        all_observation_indices.append(observation_indices)
        all_reverse_observation_indices.append(reverse_observation_indices)

    obs0 = np.array(obs0)*2
    x0 = torch.Tensor(obs0).to(device)

    mod_x0 = transform_backward(x0, all_reverse_observation_indices[0])

    all_sdes, all_ones, all_obs_ind, all_reverse_obs_ind = [], [], [], []

    end1 = time.time()
    # print("Var init time:", end1 - start_time)

    for i in range(len(policies)):
        obs_ind = torch.Tensor(all_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        reverse_obs_ind = torch.Tensor(all_reverse_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        sde, ones = diffusion_models[i].configure_sdes(num_steps=num_steps, x_T=xt[:, indices_dms[i][0]:indices_dms[i][1]], num_samples=num_samples)
        all_sdes.append(sde)
        all_ones.append(ones)
        all_obs_ind.append(obs_ind)
        all_reverse_obs_ind.append(reverse_obs_ind)

    end2 = time.time()
    # print("SDE init time:", end2 - end1)

    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        total_epsilon = torch.zeros_like(xt)

        all_epsilons = []

        end3 = time.time()
        # print("Epsilon init time:", end3 - end2)

        for i, sde, ones, indices_dm, indices_sdm, obs_ind, reverse_obs_ind, transition_model, observation_indices, reverse_observation_indices in zip(range(len(policies)), all_sdes, all_ones, indices_dms, indices_sdms, all_obs_ind, all_reverse_obs_ind, transition_models, all_observation_indices, all_reverse_observation_indices):
            end4 = time.time()

            with torch.no_grad():
                sample = xt[:, indices_dm[0]:indices_dm[1]].clone()
                sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices)
                sample[:, -state_dim:] = transform_forward(sample[:, -state_dim:], observation_indices)

                epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, obs_ind)
                
                pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
                
                pred_x0[:, -state_dim+36:] = pred_x0[:, 36:state_dim]

                epsilon = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

                epsilon[:, :state_dim] = transform_backward(epsilon[:, :state_dim], reverse_observation_indices)
                epsilon[:, -state_dim:] = transform_backward(epsilon[:, -state_dim:], reverse_observation_indices)

                total_epsilon[:, indices_dm[0]:indices_dm[1]] += epsilon

                all_epsilons.append(epsilon)

                if i > 0:
                    total_epsilon[:, indices_sdm[0]:indices_sdm[1]] = gamma[i]*all_epsilons[i-1][:, -state_dim:] + (1-gamma[i])*all_epsilons[i][:, :state_dim]
                
            end5 = time.time()
            # print("Epsilon one loop time:", end5 - end4)

        end6 = time.time()
        # print("Epsilon loop time:", end6 - end3)

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        pred_x0[:, :state_dim] = mod_x0[:state_dim]

        for i in range(len(indices_sdms)):
            pred_x0[:, indices_sdms[i][0]+12:indices_sdms[i][0]+end_index] = mod_x0[12:end_index]
            pred_x0[:, indices_sdms[i][0]+12*num_objects:indices_sdms[i][0]+state_dim] = mod_x0[12*num_objects:state_dim]

        pred_x0[:, -state_dim+12:-state_dim+end_index] = mod_x0[12:end_index]
        pred_x0[:, -state_dim+12*num_objects:] = mod_x0[12*num_objects:]

        if t > 0.25*num_steps:
            action1 = pred_x0[:, (state_dim+action_dim)+state_dim:2*(state_dim+action_dim)]
            action2 = pred_x0[:, 3*(state_dim+action_dim)+state_dim:4*(state_dim+action_dim)]
            action3 = pred_x0[:, 5*(state_dim+action_dim)+state_dim:6*(state_dim+action_dim)]

            # maximize distance between actions

            for _ in range(5):

                action1 = action1.detach()

                action1.requires_grad = True

                distance = torch.norm(action2[:, :2] - action1[:, :2], dim=1).mean()

                distance.backward()

                action1_grad = action1.grad.clone()

                action1 = action1.detach()

                action1[:, :2] = action1[:, :2] + action1_grad[:, :2]

            for _ in range(5):

                action2 = action2.detach()

                action2.requires_grad = True

                distance = torch.norm(action1[:, :2] - action2[:, :2], dim=1).mean()

                distance.backward()

                action2_grad = action2.grad.clone()

                action2 = action2.detach()

                action2[:, :2] = action2[:, :2] + action2_grad[:, :2]

            for _ in range(5):

                action3 = action3.detach()

                action3.requires_grad = True

                distance = torch.norm(action1[:, :2] - action3[:, :2], dim=1) + torch.norm(action2[:, :2] - action3[:, :2], dim=1)

                distance = distance.mean()

                distance.backward()

                action3_grad = action3.grad.clone()

                action3 = action3.detach()

                action3[:, :2] = action3[:, :2] + action3_grad[:, :2]

            pred_x0[:, 3*(state_dim+action_dim)+state_dim:4*(state_dim+action_dim)] = action2
            pred_x0[:, 5*(state_dim+action_dim)+state_dim:6*(state_dim+action_dim)] = action3
            
        with torch.no_grad():

            pred_x0 = torch.clip(pred_x0, -1, 1)

            new_epsilon = torch.randn_like(total_epsilon)

            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

        end7 = time.time()
        # print("Xt loop time:", end7 - end3)

    end8 = time.time()
    # print("Total loop time:", end8 - end2)

    xt = xt.detach().cpu().numpy()

    all_scores = []

    for i in range(1, len(indices_sdms)):
        final_states = xt[:, indices_sdms[i][0]:indices_sdms[i][1]].copy()
        scores = classifiers[i-1](torch.cat([transform_forward(torch.Tensor(final_states).to(device), all_observation_indices[i-1]), all_obs_ind[i-1]], dim=1)).detach().cpu().numpy().squeeze()
        all_scores.append(scores)

    final_states = xt[:, -state_dim:].copy()
    scores = classifiers[-1](torch.cat([transform_forward(torch.Tensor(final_states).to(device), all_observation_indices[-1]), all_obs_ind[-1]], dim=1)).detach().cpu().numpy().squeeze()
    all_scores.append(scores)

    scores = np.array(all_scores).T

    assert scores.shape == (num_samples, len(policies))

    scores = np.prod(scores, axis=1)

    assert scores.shape == (num_samples,)

    print("scores:", scores)

    sorted_indices = np.argsort(scores)[::-1][:5]

    xt = xt[sorted_indices]

    all_states = []
    all_actions = []

    for i in range(len(policies)):
        all_states.append(xt[:, indices_sdms[i][0]:indices_sdms[i][1]]*0.5)
        all_actions.append(xt[:, indices_sdms[i][1]:indices_sdms[i][1]+action_dim])
    
    all_states.append(xt[:, -state_dim:]*0.5)

    end9 = time.time()

    return all_actions, all_states

def test_dict(
    env,
    skills,
    policies,
    diffusion_models,
    transition_models,
    classifier_models,
    info_dict,
    observation_preprocessors,
    target_skill_sequence,
    device
):
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    current_skill = skills[target_skill_sequence[0]]

    policy = policies[current_skill]
    diffusion_model_transition = diffusion_models[current_skill]
    transition_model = transition_models[current_skill]
    score_model_classifier = classifier_models[current_skill]
    obs_preprocessor = observation_preprocessors[current_skill]

    diffusion_model_transition.to(device)
    transition_model.to(device)
    score_model_classifier.to(device)

    all_results = []
    all_rewards = None

    observation = info_dict["initial_observation"]
    reset_info = info_dict["reset_info"]
    target_skill_sequence = info_dict["target_skill_sequence"]
    obs0 = info_dict["obs0"]


    print("reset_info:", reset_info, "env.task", env.task)
    seed = reset_info["seed"]
    initial_observation = observation
    observation_indices = reset_info["policy_args"]["observation_indices"]
    print("primitive:", env.get_primitive(), env.action_skeleton[0].get_policy_args()["observation_indices"])
    print("reset_info:", reset_info)

    obs0 = obs_preprocessor(observation, reset_info["policy_args"])
    policy_action = query_policy_actor(policy, observation, reset_info["policy_args"])

    print("Considering skills:", skills[target_skill_sequence[0]], skills[target_skill_sequence[1]])

    actions, pred_states = get_action_from_multi_diffusion(
        policies=[policies[skills[i]] for i in target_skill_sequence],
        diffusion_models=[diffusion_models[skills[i]] for i in target_skill_sequence],
        transition_models=[transition_models[skills[i]] for i in target_skill_sequence],
        classifiers=[classifier_models[skills[i]] for i in target_skill_sequence],
        obs0=obs0.copy(),
        action_skeleton=env.action_skeleton,
        use_transition_model=False,
        num_objects=info_dict["num_objects"],
        end_index=info_dict["end_index"],
        device=device
    )

    print("observation:", observation_str(env, observation))
    print("observation tensor:", observation.shape)
    # print("action:", action_str(env, action))

    print("actions:", len(actions), actions[0].shape)
    print("policy_action:", policy_action.shape)

def evaluate_diffusion(
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    diffusion_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    closed_loop: int,
    pddl_domain: str,
    pddl_problem: str,
    max_depth: int = 5,
    timeout: float = 10.0,
    num_samples: int = 10,
    verbose: bool = False,
    load_path: Optional[Union[str, pathlib.Path]] = None,
    seed: Optional[int] = None,
    gui: Optional[int] = None,
) -> None:
    """Evaluates the policy either by loading an episode from `debug_results` or
    generating `num_eval_episodes` episodes.
    """
    if path is None and debug_results is None:
        raise ValueError("Either path or load_results must be specified")
    if seed is not None:
        random.seed(seed)

    # Load env.
    env_kwargs: Dict[str, Any] = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    if env_config is None:
        # Try to load eval env.
        env_config = pathlib.Path(checkpoint).parent / "eval/env_config.yaml"
    try:
        print("env_config:", env_config)
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None
        assert False

    all_skills = ["pick", "place", "pull", "push", "pick_hook"]
    all_policies = {}
    all_diffusion_models = {}
    all_diffusion_state_models = {}
    all_transition_models = {}
    all_classifier_models = {}
    all_observation_preprocessors = {}

    for i, policy_checkpoint, diffusion_checkpoint in zip(range(len(policy_checkpoints)), policy_checkpoints, diffusion_checkpoints):

        # Load policy.
        policy = agents.load(
            checkpoint=policy_checkpoint, env=env, env_kwargs=env_kwargs, device=device
        )

        assert isinstance(policy, agents.RLAgent)
        policy.eval_mode()

        observation_preprocessor = lambda obs, params: query_observation_vector(policy, obs, params)

        diffusion_checkpoint = pathlib.Path(diffusion_checkpoint)

        score_model_transition = ScoreNet(
            num_samples=num_samples,
            sample_dim=196,
            condition_dim=0
        )

        score_model_state = ScoreNetState(
            num_samples=num_samples,
            sample_dim=96,
            condition_dim=0
        )

        transition_model = TransitionModel(
            sample_dim=196,
            state_dim=96,
            out_channels=512
        )

        score_model_classifier = ScoreModelMLP(
            out_channels=512,
            state_dim=96,
            sample_dim=97,
        )

        diffusion_model_transition = Diffusion(
            net=score_model_transition,
        )

        diffusion_model_state = Diffusion(
            net=score_model_state,
        )

        # Load diffusion model.
        diffusion_model_transition.load_state_dict(torch.load(diffusion_checkpoint / "diffusion_model_transition.pt"))
        diffusion_model_state.load_state_dict(torch.load(diffusion_checkpoint / "diffusion_model_state.pt"))
        transition_model.load_state_dict(torch.load(diffusion_checkpoint / "transition_model.pt"))
        score_model_classifier.load_state_dict(torch.load(diffusion_checkpoint / "score_model_classifier.pt"))

        all_policies[all_skills[i]] = policy
        all_diffusion_models[all_skills[i]] = diffusion_model_transition
        all_diffusion_state_models[all_skills[i]] = diffusion_model_state
        all_transition_models[all_skills[i]] = transition_model
        all_classifier_models[all_skills[i]] = score_model_classifier
        all_observation_preprocessors[all_skills[i]] = observation_preprocessor

    # target_skill_sequence = [0, 1, 0, 1, 4, 3] #1, 0, 1, 0, 3]
    # target_skill_sequence = [4, 2, 1, 0, 1, 4, 3]
    target_skill_sequence = [4, 0, 1, 2, 4, 3]

    pkl_dicts = "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/plots/dummy_experiment/*.pkl"
    import glob
    pkl_files = sorted(glob.glob(pkl_dicts))
    
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            info_dict = pickle.load(f)
        actions, pred_states = get_action_from_multi_diffusion(
            policies=[all_policies[all_skills[i]] for i in target_skill_sequence],
            diffusion_models=[all_diffusion_models[all_skills[i]] for i in target_skill_sequence],
            transition_models=[all_transition_models[all_skills[i]] for i in target_skill_sequence],
            classifiers=[all_classifier_models[all_skills[i]] for i in target_skill_sequence],
            obs0=info_dict["obs0"].copy(),
            action_skeleton=env.action_skeleton,
            use_transition_model=False,
            num_objects=info_dict["num_objects"],
            end_index=info_dict["end_index"],
            device=device
        )

        # calculate sds loss
        all_sds_losses = []
        for i in range(len(target_skill_sequence)):
            target_skill = target_skill_sequence[i]
            observation = np.concatenate([pred_states[i], actions[i], pred_states[i+1]], axis=1)
            
            sds_loss = get_sds_loss(
                diffusion_model=all_diffusion_models[all_skills[target_skill]],
                observation=torch.Tensor(observation),
                observation_indices=env.action_skeleton[target_skill].get_policy_args()["observation_indices"],
                device=device
            )

            print(f"Skill {all_skills[target_skill]}: {sds_loss}")

            all_sds_losses.append(sds_loss)

        # geometric mean of sds losses
        sds_loss = np.prod(all_sds_losses)**(1/len(all_sds_losses))
        print(f"Total SDS Loss: {sds_loss} for {target_skill_sequence}")

        assert False

        all_predicted_initial_states = []
        all_predicted_final_states = []
        for sk in range(5):
            id1 = sk
            id2 = sk if sk < 4 else 0
            all_skill_states = forward_diffusion(
                diffusion_model=all_diffusion_models[all_skills[id1]],
                observation=info_dict["obs0"] if id1 == target_skill_sequence[0] else None,
                observation_indices=env.action_skeleton[id2].get_policy_args()["observation_indices"],
                reset_observation_indices=env.action_skeleton[target_skill_sequence[0]].get_policy_args()["observation_indices"],
                num_samples=100,
                num_objects=info_dict["num_objects"],
                end_index=info_dict["end_index"],
                num_steps=256,
                device=device
            )

            all_skill_states = np.array(all_skill_states)
            all_predicted_initial_states.append(all_skill_states[:, -1, :96])
            all_predicted_final_states.append(all_skill_states[:, -1, -96:])    

        all_distances = []

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, len(target_skill_sequence)-1, figsize=(5*(len(target_skill_sequence)-1), 5))

        # plt.hist(diff1, bins=100, alpha=0.5, label='Distribution 1')
        # plt.hist(diff2, bins=100, alpha=0.5, label='Distribution 2')
        # plt.legend()
        # plt.savefig('distance_histogram1.png')

        skeleton_score = 0

        for j in range(len(target_skill_sequence)-1):
            skill1 = target_skill_sequence[j]
            skill2 = target_skill_sequence[j+1]

            min_dist1, min_dist2, score = generate_distance_matrix(
                [all_predicted_final_states[skill1], all_predicted_initial_states[skill2]],
                lambda x: transform_forward(x, env.action_skeleton[skill2].get_policy_args()["observation_indices"]),
            )
            
            skeleton_score += score

            ax[j].hist(min_dist1, bins=100, alpha=0.5, label='Distribution 1')
            ax[j].hist(min_dist2, bins=100, alpha=0.5, label='Distribution 2')
            ax[j].legend()
            ax[j].set_title(f"Skill {all_skills[skill1]} to Skill {all_skills[skill2]}")

        skeleton_score /= len(target_skill_sequence)-1

        named_skills = [all_skills[i] for i in target_skill_sequence]
        # position the title at the bottom center
        plt.suptitle(
            f"Total Score: {skeleton_score} for {named_skills} infeasible conditional",   
        )
        plt.tight_layout()
        plt.savefig(f'distance_histogram_{target_skill_sequence}_infeasible_conditional.png')

        assert False
            
@torch.no_grad()
def generate_distance_matrix(all_samples, encoder=None):

    dist1 = all_samples[0]
    dist2 = all_samples[1]

    if encoder is not None:
        dist1 = encoder(dist1)
        dist2 = encoder(dist2)

    # calculate chamfer distance between dist1 and dist2
    # dist1 = dist1.unsqueeze(1).cpu().numpy()
    # dist2 = dist2.unsqueeze(0).cpu().numpy()
    dist1 = dist1[:, None, :36]
    dist2 = dist2[None, :, :36]
    diff = dist1 - dist2
    diff = np.linalg.norm(diff[:, :, 12:18], axis=-1) + np.linalg.norm(diff[:, :, 24:30], axis=-1)

    print("diff:", diff.shape)

    min_dist1 = np.min(diff, axis=1)
    min_dist2 = np.min(diff, axis=0)

    return min_dist1, min_dist2, np.mean(min_dist1) + np.mean(min_dist2)

@torch.no_grad()
def get_sds_loss(
    diffusion_model: Diffusion,
    observation: torch.Tensor,
    observation_indices: np.ndarray,
    device: torch.device = "auto",
    num_steps: int = 256,
    state_dim: int = 96,
):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    diffusion_model.to(device)

    num_samples = observation.shape[0]
    obs_ind = torch.Tensor(observation_indices).to(device).unsqueeze(0).repeat(num_samples, 1)

    # all_samples.append(observation.clone().unsqueeze(0).cpu().numpy())

    observation[:, :state_dim] = transform_forward(observation[:, :state_dim], observation_indices)
    observation[:, -state_dim:] = transform_forward(observation[:, -state_dim:], observation_indices)

    xt = torch.zeros_like(observation).to(device)

    sde, ones = diffusion_model.configure_sdes(num_steps=num_steps, x_T=xt, num_samples=num_samples)

    all_grads = []
    for t in range(num_steps):

        alpha_t = sde.base_sde.alpha(t * ones/sde.N).to(device)
        noise = torch.randn_like(observation)
        observation = observation.to(device)
        noise = noise.to(device)

        sample = torch.sqrt(alpha_t) * observation + torch.sqrt(1 - alpha_t) * noise
        epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, obs_ind)

        error_state1 = (epsilon - noise)[:, :state_dim][:, 12:36]
        error_state2 = (epsilon - noise)[:, -state_dim:][:, 12:36]

        w = 1 - torch.sqrt(alpha_t)
        grad = w * (torch.linalg.norm(error_state1, dim=1) + torch.linalg.norm(error_state2, dim=1))
        all_grads.append(grad.mean().item())

    return np.mean(all_grads)

def main(args: argparse.Namespace) -> None:
    evaluate_diffusion(**vars(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument(
        "--diffusion-checkpoints", "-c", nargs="+", help="Diffusion checkpoints"
    )
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument(
        "--closed-loop", default=1, type=int, help="Run closed-loop planning"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--pddl-domain", help="Pddl domain", default=None)
    parser.add_argument("--pddl-problem", help="Pddl problem", default=None)
    parser.add_argument(
        "--max-depth", type=int, default=4, help="Task planning search depth"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Task planning timeout"
    )
    args = parser.parse_args()

    args.env_config = pathlib.Path("/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/configs/pybullet/envs/official/domains/rearrangement_push/task1.yaml")
    args.policy_checkpoints = [
       "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/models/20221024/decoupled_state/pick/ckpt_model_200000.pt",
       "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/models/20221024/decoupled_state/place/ckpt_model_200000.pt",
       "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/models/20221024/decoupled_state/pull/ckpt_model_200000.pt",
       "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/models/20221024/decoupled_state/push/ckpt_model_200000.pt",
       "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/models/20221024/decoupled_state/pick/ckpt_model_200000.pt",
    ]
    args.diffusion_checkpoints = [
        "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/diffusion_models/v7_final_1/unnormalized_pick/",
        "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/diffusion_models/v7_final_1/unnormalized_place/",
        "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/diffusion_models/v7_final_1/unnormalized_pull/",
        "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/diffusion_models/v7_final_1/unnormalized_push/",
        "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/diffusion_models/v7_final_1/unnormalized_pick_hook/"
    ]
    args.path = "/home/umishra31/Fall2024/GSC_task/gsc_suite/temporal_policies/plots/dummy_experiment/"
    args.device = "cuda"
    args.num_eval = 50
    args.closed_loop = 1
    args.seed = 0
    args.gui = 0
    args.verbose = 1
    args.pddl_domain = None
    args.pddl_problem = None
    args.max_depth = 4
    args.timeout = 10.0

    main(args)