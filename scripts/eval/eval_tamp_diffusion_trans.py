#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import symbolic
import tqdm
import json
from PIL import Image

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
    gamma: Sequence[float] = [1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    device: torch.device = "auto"
) -> np.ndarray:

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

    for i in range(len(policies)):
        obs_ind = torch.Tensor(all_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        reverse_obs_ind = torch.Tensor(all_reverse_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        sde, ones = diffusion_models[i].configure_sdes(num_steps=num_steps, x_T=xt[:, indices_dms[i][0]:indices_dms[i][1]], num_samples=num_samples)
        all_sdes.append(sde)
        all_ones.append(ones)
        all_obs_ind.append(obs_ind)
        all_reverse_obs_ind.append(reverse_obs_ind)

    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        total_epsilon = torch.zeros_like(xt)

        all_epsilons = []

        for i, sde, ones, indices_dm, indices_sdm, obs_ind, reverse_obs_ind, transition_model, observation_indices, reverse_observation_indices in zip(range(len(policies)), all_sdes, all_ones, indices_dms, indices_sdms, all_obs_ind, all_reverse_obs_ind, transition_models, all_observation_indices, all_reverse_observation_indices):

            with torch.no_grad():
                sample = xt[:, indices_dm[0]:indices_dm[1]].clone()
                sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices)
                sample[:, -state_dim:] = transform_forward(sample[:, -state_dim:], observation_indices)

                epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, obs_ind)
                
                pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
                
                # if i == 0:
                #     pred_x0[:, :state_dim] = x0

                pred_x0[:, -state_dim+36:] = pred_x0[:, 36:state_dim]

            # if i == 0:
            #     current_state = pred_x0[:, :state_dim].detach()
            #     current_action = pred_x0[:, state_dim:state_dim+action_dim].detach()
            #     for _ in range(5):
            #         current_action = current_action.clone()
            #         current_action.requires_grad = True
            #         next_state = transition_model(torch.cat([current_state, current_action, obs_ind], dim=1))
            #         target_next_state = pred_x0[:, state_dim+action_dim:]
            #         loss = F.mse_loss(next_state, target_next_state)
            #         loss.backward()
            #         current_action = current_action.detach() - current_action.grad.detach()
            #     pred_x0[:, state_dim:state_dim+action_dim] = current_action
            
            with torch.no_grad():

                if i == 3 or i == 5:
                    pred_x0[:, state_dim:state_dim+action_dim] = policies[i].actor.predict(pred_x0[:, :state_dim])

                if use_transition_model: # or i % 2 == 0:
                    pred_x0[:, state_dim+action_dim:] = transition_model(torch.cat([pred_x0[:, :state_dim+action_dim], obs_ind], dim=1))

                pred_x0 = torch.clip(pred_x0, -1, 1)

                epsilon = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

                epsilon[:, :state_dim] = transform_backward(epsilon[:, :state_dim], reverse_observation_indices)
                epsilon[:, -state_dim:] = transform_backward(epsilon[:, -state_dim:], reverse_observation_indices)

                total_epsilon[:, indices_dm[0]:indices_dm[1]] += epsilon

                all_epsilons.append(epsilon)

                if i > 0:
                    total_epsilon[:, indices_sdm[0]:indices_sdm[1]] = gamma[i]*all_epsilons[i-1][:, -state_dim:] + (1-gamma[i])*all_epsilons[i][:, :state_dim]

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        # pred_x0[:, :state_dim] = transform_backward(pred_x0[:, :state_dim], reverse_observation_indices)
        # pred_x0[:, state_dim+action_dim:] = transform_backward(pred_x0[:, state_dim+action_dim:], reverse_observation_indices)

        pred_x0[:, :state_dim] = mod_x0[:state_dim]

        if t > 0.8*num_steps:
            pred_x0[:, -state_dim:] = mod_x0[:state_dim]

        for i in range(len(indices_sdms)):
            pred_x0[:, indices_sdms[i][0]+12:indices_sdms[i][0]+end_index] = mod_x0[12:end_index]
            pred_x0[:, indices_sdms[i][0]+12*num_objects:indices_sdms[i][0]+state_dim] = mod_x0[12*num_objects:state_dim]

        pred_x0[:, -state_dim+12:-state_dim+end_index] = mod_x0[12:end_index]
        pred_x0[:, -state_dim+12*num_objects:] = mod_x0[12*num_objects:]

        # pick and place actions are same for 1st and 3rd policy

        # pred_x0[:, 3*(state_dim+action_dim)+24:3*(state_dim+action_dim)+48] = pred_x0[:, 24:48]

        if t > 0.25*num_steps:
            action_place1 = pred_x0[:, 2*(state_dim+action_dim)+state_dim:3*(state_dim+action_dim)]
            action_place2 = pred_x0[:, 4*(state_dim+action_dim)+state_dim:5*(state_dim+action_dim)]
            
            # for _ in range(5):

            #     action_place2 = action_place2.clone()

            #     action_place2.requires_grad = True

            #     # maximize distance from origin

            #     distance = torch.norm(action_place2[:, :2], dim=1, keepdim=True)
            #     distance = distance.mean()
            #     distance.backward()

            #     action_place2_grad = action_place2.grad.detach()

            #     action_place2 = action_place2.detach()

            #     action_place2[:, :2] = action_place2[:, :2] + action_place2_grad[:, :2]

            # for _ in range(5):

            #     action_place1 = action_place1.clone()

            #     action_place1.requires_grad = True

            #     # maximize distance from origin

            #     distance = torch.norm(action_place1[:, :1] - action_place2[:, :1], dim=1, keepdim=True)
            #     distance = distance.mean()
            #     distance.backward()

            #     action_place1_grad = action_place1.grad.detach()

            #     action_place1 = action_place1.detach()

            #     action_place1[:, :1] = action_place1[:, :1] - 0.1*action_place1_grad[:, :1]

            pred_x0[:, 2*(state_dim+action_dim)+state_dim:3*(state_dim+action_dim)] = action_place1
            pred_x0[:, 4*(state_dim+action_dim)+state_dim:5*(state_dim+action_dim)] = action_place2

            
        with torch.no_grad():

            pred_x0 = torch.clip(pred_x0, -1, 1)

            new_epsilon = torch.randn_like(total_epsilon)

            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

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

    return all_actions, all_states

def get_action_from_two_diffusion(
    policy1,
    policy2,
    diffusion_model1: Diffusion,
    diffusion_model2: Diffusion,
    transition_model1: TransitionModel,
    transition_model2: TransitionModel,
    classifier1: ScoreModelMLP,
    classifier2: ScoreModelMLP,
    obs0: torch.Tensor,
    observation_indices1: np.ndarray,
    observation_indices2: np.ndarray,
    use_transition_model: bool = True,
    num_samples: int = 5,
    num_objects: int = 4,
    state_dim: int = 96,
    action_dim: int = 4,
    gamma: float = 0.0,
    device: torch.device = "auto"
) -> np.ndarray:

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    diffusion_model1.to(device)
    diffusion_model2.to(device)
    transition_model1.to(device)
    transition_model2.to(device)
    classifier1.to(device)
    classifier2.to(device)

    num_steps = 256
    sample_dim = state_dim + action_dim + state_dim + action_dim + state_dim

    xt = torch.zeros(num_samples, sample_dim).to(device)

    reverse_observation_indices1 = np.zeros_like(observation_indices1)
    reverse_observation_indices2 = np.zeros_like(observation_indices2)

    for i in range(len(observation_indices1)):
        reverse_observation_indices1[observation_indices1[i]] = i

    for i in range(len(observation_indices2)):
        reverse_observation_indices2[observation_indices2[i]] = i

    obs0 = np.array(obs0)*2
    x0 = torch.Tensor(obs0).to(device)

    obs_ind1 = torch.Tensor(observation_indices1).to(device).unsqueeze(0).repeat(num_samples, 1)
    obs_ind2 = torch.Tensor(observation_indices2).to(device).unsqueeze(0).repeat(num_samples, 1)

    rev_obs_ind1 = torch.Tensor(reverse_observation_indices1).to(device).unsqueeze(0).repeat(num_samples, 1)
    rev_obs_ind2 = torch.Tensor(reverse_observation_indices2).to(device).unsqueeze(0).repeat(num_samples, 1)

    sde1, ones1 = diffusion_model1.configure_sdes(num_steps=num_steps, x_T=xt[:, :state_dim+action_dim+state_dim], num_samples=num_samples)
    sde2, ones2 = diffusion_model2.configure_sdes(num_steps=num_steps, x_T=xt[:, state_dim+action_dim:], num_samples=num_samples)

    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        total_epsilon = torch.zeros_like(xt)

        sample = xt[:, :state_dim+action_dim+state_dim].clone()
        sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices1)
        sample[:, state_dim+action_dim:state_dim+action_dim+state_dim] = transform_forward(sample[:, state_dim+action_dim:state_dim+action_dim+state_dim], observation_indices1)

        epsilon1, alpha_t, alpha_tm1 = sde1.sample_epsilon(t * ones1, sample, obs_ind1)
        
        pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon1) / torch.sqrt(alpha_t)
        
        pred_x0[:, :state_dim] = x0
        if use_transition_model:
            pred_x0[:, state_dim+action_dim:] = transition_model1(torch.cat([pred_x0[:, :state_dim+action_dim], obs_ind1], dim=1))

        pred_x0 = torch.clip(pred_x0, -1, 1)

        epsilon1 = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

        epsilon1[:, :state_dim] = transform_backward(epsilon1[:, :state_dim], reverse_observation_indices1)
        epsilon1[:, state_dim+action_dim:state_dim+action_dim+state_dim] = transform_backward(epsilon1[:, state_dim+action_dim:state_dim+action_dim+state_dim], reverse_observation_indices1)

        total_epsilon[:, :state_dim+action_dim+state_dim] += epsilon1

        sample = xt[:, state_dim+action_dim:].clone()
        sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices2)
        sample[:, state_dim+action_dim:state_dim+action_dim+state_dim] = transform_forward(sample[:, state_dim+action_dim:state_dim+action_dim+state_dim], observation_indices2)

        epsilon2, alpha_t, alpha_tm1 = sde2.sample_epsilon(t * ones2, sample, obs_ind2)
        
        pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon2) / torch.sqrt(alpha_t)
        
        if use_transition_model:
            pred_x0[:, state_dim+action_dim:] = transition_model2(torch.cat([pred_x0[:, :state_dim+action_dim], obs_ind2], dim=1))

        pred_x0 = torch.clip(pred_x0, -1, 1)

        epsilon2 = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

        epsilon2[:, :state_dim] = transform_backward(epsilon2[:, :state_dim], reverse_observation_indices2)
        epsilon2[:, state_dim+action_dim:state_dim+action_dim+state_dim] = transform_backward(epsilon2[:, state_dim+action_dim:state_dim+action_dim+state_dim], reverse_observation_indices2)

        total_epsilon[:, state_dim+action_dim:] += epsilon2

        total_epsilon[:, state_dim+action_dim:state_dim+action_dim+state_dim] = gamma*epsilon1[:, -state_dim:] + (1-gamma)*epsilon2[:, :state_dim]

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        # pred_x0[:, :state_dim] = transform_backward(pred_x0[:, :state_dim], reverse_observation_indices)
        # pred_x0[:, state_dim+action_dim:] = transform_backward(pred_x0[:, state_dim+action_dim:], reverse_observation_indices)

        mod_x0 = transform_backward(x0, reverse_observation_indices1)
        pred_x0[:, state_dim+action_dim+12:state_dim+action_dim+24] = mod_x0[12:24]
        pred_x0[:, state_dim+action_dim+12*num_objects:state_dim+action_dim+state_dim] = mod_x0[12*num_objects:state_dim]

        pred_x0[:, state_dim+action_dim+state_dim+action_dim+12:state_dim+action_dim+state_dim+action_dim+24] = mod_x0[12:24]
        pred_x0[:, state_dim+action_dim+state_dim+action_dim+12*num_objects:state_dim+action_dim+state_dim+action_dim+state_dim] = mod_x0[12*num_objects:state_dim]
        
        new_epsilon = torch.randn_like(total_epsilon)

        xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

    xt = xt.detach().cpu().numpy()

    state1 = xt[:, :state_dim]
    action1 = xt[:, state_dim:state_dim+action_dim]
    state2 = xt[:, state_dim+action_dim:state_dim+action_dim+state_dim]
    action2 = xt[:, state_dim+action_dim+state_dim:state_dim+action_dim+state_dim+action_dim]
    state3 = xt[:, state_dim+action_dim+state_dim+action_dim:]

    scores = classifier2(torch.cat([transform_forward(torch.Tensor(state3).to(device), observation_indices2), obs_ind2], dim=1)).detach().cpu().numpy().squeeze()

    print("scores:", scores)

    sorted_indices = np.argsort(scores)[::-1]

    xt = xt[sorted_indices]

    state1 = xt[:, :state_dim]*0.5
    action1 = xt[:, state_dim:state_dim+action_dim]
    state2 = xt[:, state_dim+action_dim:state_dim+action_dim+state_dim]*0.5
    action2 = xt[:, state_dim+action_dim+state_dim:state_dim+action_dim+state_dim+action_dim]
    state3 = xt[:, state_dim+action_dim+state_dim+action_dim:]*0.5

    return [action1, action2], [state1, state2, state3]

def get_action_from_diffusion(
    policy,
    diffusion_model: Diffusion,
    transition_model: TransitionModel,
    classifier: ScoreModelMLP,
    obs0: torch.Tensor,
    observation_indices: np.ndarray,
    use_transition_model: bool = True,
    num_samples: int = 5,
    state_dim: int = 96,
    action_dim: int = 4,
    num_objects: int = 4,
    device: torch.device = "auto"
) -> np.ndarray:
    """Samples an action from the diffusion model."""
    # Sample action from diffusion model.

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    num_steps = 256
    sample_dim = state_dim + action_dim + state_dim

    xt = torch.zeros(num_samples, sample_dim).to(device)

    reverse_observation_indices = np.zeros_like(observation_indices)

    for i in range(len(observation_indices)):
        reverse_observation_indices[observation_indices[i]] = i

    obs0 = np.array(obs0)*2
    x0 = torch.Tensor(obs0).to(device)

    obs_ind = torch.Tensor(observation_indices).to(device).unsqueeze(0).repeat(num_samples, 1)
    rev_obs_ind = torch.Tensor(reverse_observation_indices).to(device).unsqueeze(0).repeat(num_samples, 1)

    sde, ones = diffusion_model.configure_sdes(num_steps=num_steps, x_T=xt, num_samples=num_samples)

    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        sample = xt.clone()
        sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices)
        sample[:, state_dim+action_dim:] = transform_forward(sample[:, state_dim+action_dim:], observation_indices)

        epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, obs_ind)
        
        pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
        
        pred_x0[:, :state_dim] = x0
        if use_transition_model:
            pred_x0[:, state_dim+action_dim:] = transition_model(torch.cat([pred_x0[:, :state_dim+action_dim], obs_ind], dim=1))

        pred_x0 = torch.clip(pred_x0, -1, 1)

        epsilon = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

        new_epsilon = torch.randn_like(epsilon)

        pred_x0[:, :state_dim] = transform_backward(pred_x0[:, :state_dim], reverse_observation_indices)
        pred_x0[:, state_dim+action_dim:] = transform_backward(pred_x0[:, state_dim+action_dim:], reverse_observation_indices)

        mod_x0 = transform_backward(x0, reverse_observation_indices)
        pred_x0[:, state_dim+action_dim+12:state_dim+action_dim+24] = mod_x0[12:24]
        pred_x0[:, state_dim+action_dim+12*num_objects:state_dim+action_dim+state_dim] = mod_x0[12*num_objects:state_dim]
        
        xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

    xt = xt.detach().cpu().numpy()

    initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    scores = classifier(torch.cat([torch.Tensor(initial_state).to(device), obs_ind], dim=1)).detach().cpu().numpy().squeeze()

    print("scores:", scores)

    sorted_indices = np.argsort(scores)[::-1]

    xt = xt[sorted_indices]

    initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    return action, [initial_state*0.5, final_state*0.5]

def evaluate_episodes(
    env: envs.Env,
    skills: Sequence[str] = [],
    policies: Optional[Sequence[Optional[agents.RLAgent]]] = None,
    diffusion_models: Optional[Sequence[Optional[Diffusion]]] = None,
    diffusion_state_models: Optional[Sequence[Optional[Diffusion]]] = None,
    transition_models: Optional[Sequence[Optional[TransitionModel]]] = None,
    classifier_models: Optional[Sequence[Optional[ScoreModelMLP]]] = None,
    observation_preprocessors: Optional[Sequence[Optional[Callable]]] = None,
    target_skill_sequence: Sequence[int] = [],
    target_length: int = 10,
    num_episodes: int = 5,
    path: Optional[Union[str, pathlib.Path]] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
    device: str = "auto",
) -> None:
    """Evaluates policy for the given number of episodes."""
    num_successes = 0
    pbar = tqdm.tqdm(
        range(num_episodes),
        desc=f"Evaluate {env.name}",
        dynamic_ncols=True,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    current_skill = skills[target_skill_sequence[0]]

    policy = policies[current_skill]
    diffusion_model_transition = diffusion_models[current_skill]
    diffusion_model_state = diffusion_state_models[current_skill]
    transition_model = transition_models[current_skill]
    score_model_classifier = classifier_models[current_skill]
    obs_preprocessor = observation_preprocessors[current_skill]

    diffusion_model_transition.to(device)
    diffusion_model_state.to(device)
    transition_model.to(device)
    score_model_classifier.to(device)

    all_results = []

    if not path.exists():
        path.mkdir(parents=True)

    # remove all gif files from path
    for f in path.iterdir():
        if f.suffix == ".gif" or f.suffix == ".png":
            f.unlink()

    for ep in pbar:
        # Evaluate episode.
        observation, reset_info = env.reset() #seed=seed)
        print("reset_info:", reset_info, "env.task", env.task)
        seed = reset_info["seed"]
        initial_observation = observation
        observation_indices = reset_info["policy_args"]["observation_indices"]
        print("primitive:", env.get_primitive(), env.action_skeleton[0].get_policy_args()["observation_indices"])
        print("reset_info:", reset_info)

        rewards = []
        done = False

        obs0 = obs_preprocessor(observation, reset_info["policy_args"])
        policy_action = query_policy_actor(policy, observation, reset_info["policy_args"])

        target_skill_sequence = target_skill_sequence[:target_length]

        # actions, pred_states = get_action_from_diffusion(
        #     policy,
        #     diffusion_model_transition,
        #     transition_model,
        #     score_model_classifier,
        #     obs0,
        #     observation_indices=observation_indices,
        #     use_transition_model=False,
        #     device=device
        # )

        print("Considering skills:", skills[target_skill_sequence[0]], skills[target_skill_sequence[1]])

        # actions, pred_states = get_action_from_two_diffusion(
        #     policies[skills[target_skill_sequence[0]]],
        #     policies[skills[target_skill_sequence[1]]],
        #     diffusion_models[skills[target_skill_sequence[0]]],
        #     diffusion_models[skills[target_skill_sequence[1]]],
        #     transition_models[skills[target_skill_sequence[0]]],
        #     transition_models[skills[target_skill_sequence[1]]],
        #     classifier_models[skills[target_skill_sequence[0]]],
        #     classifier_models[skills[target_skill_sequence[1]]],
        #     obs0,
        #     observation_indices1=np.array(env.action_skeleton[0].get_policy_args()["observation_indices"]),
        #     observation_indices2=np.array(env.action_skeleton[1].get_policy_args()["observation_indices"]),
        #     use_transition_model=False,
        #     device=device
        # )

        actions, pred_states = get_action_from_multi_diffusion(
            policies=[policies[skills[i]] for i in target_skill_sequence],
            diffusion_models=[diffusion_models[skills[i]] for i in target_skill_sequence],
            transition_models=[transition_models[skills[i]] for i in target_skill_sequence],
            classifiers=[classifier_models[skills[i]] for i in target_skill_sequence],
            obs0=obs0,
            action_skeleton=env.action_skeleton,
            use_transition_model=False,
            num_objects=5,
            end_index=36,
            device=device
        )

        if verbose:
            print("observation:", observation_str(env, observation))
            print("observation tensor:", observation)
            # print("action:", action_str(env, action))

        print("actions:", actions)
        print("policy_action:", policy_action)

        for j in range(actions[0].shape[0]):

            env.reset(seed=seed)
            env.set_observation(initial_observation)

            env.record_start()

            for i, action in enumerate(actions):

                env.set_primitive(env.action_skeleton[i])

                try:
                    # action = policy_action
                    observation, reward, terminated, truncated, step_info = env.step(action[j])
                except Exception as e:
                    print("Exception:", e)
                    assert False, "Exception occurred"
                    continue

                if verbose:
                    print("step_info:", step_info)
                    
                print(f"Action for: {skills[target_skill_sequence[i]]}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")

                rewards.append(reward)
                done = terminated or truncated

            success = np.prod(rewards) > 0

            env.record_stop()

            if success:
                env.record_save(path / f"eval_{ep}_{i}_{j}_success.gif", reset=True)

                imgs = []

                for state in pred_states:
                    curr_state = state[j].reshape(8, 12)
                    curr_state = policy.encoder.unnormalize(torch.Tensor(curr_state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                    env.set_observation(curr_state)
                    imgs.append(env.render())

                imgs = np.concatenate(imgs, axis=1)

                Image.fromarray(imgs).save(path / f"eval_{ep}_{i}_{j}_success.png")              

            else:
                env.record_save(path / f"eval_{ep}_{i}_{j}_fail.gif", reset=True)

                imgs = []

                for state in pred_states:
                    curr_state = state[j].reshape(8, 12)
                    curr_state = policy.encoder.unnormalize(torch.Tensor(curr_state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                    env.set_observation(curr_state)
                    imgs.append(env.render())

                imgs = np.concatenate(imgs, axis=1)

                Image.fromarray(imgs).save(path / f"eval_{ep}_{i}_{j}_fail.png")   

            if success:
                rewards = []
                env.set_observation(initial_observation)
            else:
                rewards = []
                env.set_observation(initial_observation)

        num_successes += success
        pbar.set_postfix(
            {"rewards": rewards, "successes": f"{num_successes} / {num_episodes}"}
        )

    # save results as json
    # with open(path / f"results_{seed}.json", "w") as f:
    #     json.dump(
    #         {
    #             "num_episodes": num_episodes,
    #             "num_successes": num_successes,
    #         }, f)



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

    all_skills = ["pick", "place", "pull", "push"]
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

    target_skill_sequence = [0, 2, 1, 0, 1, 0, 3]
    target_length = len(target_skill_sequence)
    num_episodes = 5

    evaluate_episodes(
        env=env,
        skills=all_skills,
        policies=all_policies,
        diffusion_models=all_diffusion_models,
        diffusion_state_models=all_diffusion_state_models,
        transition_models=all_transition_models,
        classifier_models=all_classifier_models,
        observation_preprocessors=all_observation_preprocessors,
        target_skill_sequence=target_skill_sequence,
        target_length=target_length,
        num_episodes=num_episodes,
        path=pathlib.Path(path),
        verbose=verbose,
        seed=seed,
    )


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

    main(args)
