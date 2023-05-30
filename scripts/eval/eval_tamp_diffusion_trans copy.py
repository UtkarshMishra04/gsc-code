import argparse
import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import symbolic
import tqdm
from PIL import Image

from temporal_policies import dynamics, envs, planners, agents
from temporal_policies.envs.pybullet.table import primitives as table_primitives
from temporal_policies.utils import recording, timing

from temporal_policies.diff_models.unet_transformer import ScoreNet, ScoreNetState
from temporal_policies.diff_models.classifier_transformer import ScoreModelMLP #, TransitionModel
from temporal_policies.diff_models.classifier import TransitionModel
from temporal_policies.mixed_diffusion.cond_diffusion1D import Diffusion

def action_str(env: envs.Env, action: np.ndarray) -> str:
    """Converts actions to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        primitive = env.get_primitive()
        assert isinstance(primitive, envs.pybullet.table.primitives.Primitive)
        return str(primitive.Action(action))

    return str(action)

def seed_generator(
    num_eval: int,
    path_results: Optional[Union[str, pathlib.Path]] = None,
) -> Generator[
    Tuple[
        Optional[int],
        Optional[Tuple[np.ndarray, planners.PlanningResult, Optional[List[float]]]],
    ],
    None,
    None,
]:
    if path_results is not None:
        npz_files = sorted(
            pathlib.Path(path_results).glob("results_*.npz"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        for npz_file in npz_files:
            with open(npz_file, "rb") as f:
                npz = np.load(f, allow_pickle=True)
                seed: int = npz["seed"].item()
                rewards = np.array(npz["rewards"])
                plan = planners.PlanningResult(
                    actions=np.array(npz["actions"]),
                    states=np.array(npz["states"]),
                    p_success=npz["p_success"].item(),
                    values=np.array(npz["values"]),
                )
                t_planner: List[float] = npz["t_planner"].item()

            yield seed, (rewards, plan, t_planner)

    if num_eval is not None:
        yield 0, None
        for _ in range(num_eval - 1):
            yield None, None


def scale_actions(
    actions: np.ndarray,
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
) -> np.ndarray:
    scaled_actions = actions.copy()
    for t, primitive in enumerate(action_skeleton):
        action_dims = primitive.action_space.shape[0]
        scaled_actions[..., t, :action_dims] = primitive.scale_action(
            actions[..., t, :action_dims]
        )

    return scaled_actions


def task_plan(
    pddl: symbolic.Pddl,
    env: envs.Env,
    max_depth: int = 10,
    timeout: float = 10.0,
    verbose: bool = False,
) -> Generator[List[envs.Primitive], None, None]:
    planner = symbolic.Planner(pddl, pddl.initial_state)
    max_depth = 10
    bfs = symbolic.BreadthFirstSearch(
        planner.root, max_depth=max_depth, timeout=timeout, verbose=False
    )
    for plan in bfs:
        print("plan:", plan)
        action_skeleton = [
            env.get_primitive_info(action_call=str(node.action)) for node in plan[1:]
        ]
        yield action_skeleton

def query_policy_actor(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    # print("policy.encoder:", policy.encoder)
    # print("policy.encoder.encode(observation.to(policy.device), policy_args):", policy.encoder.encode(observation.to(policy.device), policy_args))
    return policy.actor.predict(
        policy.encoder.encode(observation.to(policy.device), policy_args)
    )

def query_observation_vector(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.encoder.encode(observation.to(policy.device), policy_args)


def transform_observation_from_one_policy_to_another(
    observation: torch.Tensor,
    policy_from: agents.RLAgent,
    policy_to: agents.RLAgent,
    action_primitive_to: envs.Primitive,
    env: envs.Env,
) -> torch.Tensor:

    current_primitive = env.get_primitive()
    gt_observation = policy_from.encoder.decode(observation, current_primitive.get_policy_args())

    env.set_primitive(action_primitive_to)

    new_observation = policy_to.encoder.encode(gt_observation, action_primitive_to.get_policy_args())

    # observation_indices = np.array(current_primitive.get_policy_args()["observation_indices"])
 
    # reverse_observation_indices = np.zeros_like(observation_indices)

    # print("observation_indices:", observation_indices, "reverse_observation_indices:", reverse_observation_indices)

    # for i in range(len(reverse_observation_indices)):
    #     print("i:", i, "np.where(observation_indices == i):", np.where(observation_indices == i))
    #     reverse_observation_indices[i] = np.where(observation_indices == i)[0][0]

    # observation_indices = action_primitive_to.get_policy_args()["observation_indices"]

    # check_obs_1 = observation.reshape(8,12)[reverse_observation_indices][observation_indices].reshape(1,-1)

    # assert torch.norm(check_obs_1 - new_observation) < 1e-3

    # new_gt_observation = policy_to.encoder.decode(new_observation, action_primitive_to.get_policy_args())
    
    return new_observation

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

def modify_transition_gradient(
    current_state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    transition_model: TransitionModel,
    observation_indices: torch.Tensor,
    num_grad_steps: int = 50
):
    ############################################
    # Perform the last step discrimination (refinement)
    # candidate_samples: [batch_size, num_samples, sample_dim]
    # returns: [batch_size, num_samples, sample_dim]
    ############################################

    state_dim = current_state.shape[1]
    action_dim = action.shape[1]

    device = current_state.device
    current_state = current_state.detach()
    action = action.detach()
    next_state = next_state.detach()
    observation_indices = observation_indices.detach()

    prev_loss = 0

    for i in range(num_grad_steps):

        # current_vector = torch.cat([current_state, action, observation_indices], dim=1)
        current_vector = torch.cat([current_state, action], dim=1)

        current_vector = current_vector.clone()
        current_vector.requires_grad = True

        predicted_next_state = transition_model(current_vector)

        loss = torch.nn.MSELoss()(predicted_next_state, next_state)

        loss = loss.mean()

        loss.backward()

        action_grads = 0.01 * current_vector.grad[:, state_dim:state_dim+action_dim] + 0.01 * torch.randn_like(action)

        action = action - action_grads

        action = action.detach()

        prev_loss = loss.item()        

    return action

def modify_gradient_classifier(
    samples: torch.Tensor,
    obs_ind: torch.Tensor,
    score_model: ScoreModelMLP,
    target: int,
    num_grad_steps: int = 10
):
    ############################################
    # Perform the last step discrimination (refinement)
    # candidate_samples: [batch_size, num_samples, sample_dim]
    # returns: [batch_size, num_samples, sample_dim]
    ############################################

    device = samples.device
    samples = samples.detach()

    prev_loss = 0

    for i in range(num_grad_steps):

        samples = samples.clone()
        samples.requires_grad = True

        predicted_score = score_model(torch.cat([samples, obs_ind], dim=1))

        loss = torch.nn.BCELoss()(predicted_score, torch.ones_like(predicted_score)*target)

        loss = loss.mean()

        loss.backward()

        samples = samples - 0.05 * samples.grad - 0.01 * torch.randn_like(samples)

        samples = samples.detach()

        prev_loss = loss.item()        

    return samples

def plan(
    policies: Sequence[agents.RLAgent],
    diffusion_models: Sequence[Diffusion],
    diffusion_state_models: Sequence[Diffusion],
    transition_models: Sequence[TransitionModel],
    score_model_classifiers: Sequence[ScoreModelMLP],
    action_skeleton: Sequence[envs.Primitive],
    observation: np.ndarray,
    observation_indices: np.ndarray,
    num_samples: int = 20,
    device: str = "auto",
    closed_loop: bool = False,
    gamma1: float = 0.2,
    gamma2: float = 0.0,
    target_encoded_obs: torch.Tensor = None,
):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    print("action_skeleton:", action_skeleton)
    print("observation:", observation)
    print("policies:", len(policies))
    print("diffusion_models:", len(diffusion_models))

    all_observation_indices = []
    all_reverse_observation_indices = []

    for primitive in action_skeleton:
        observation_indices = np.array(primitive.get_policy_args()["observation_indices"])
        all_observation_indices.append(observation_indices)
        
        reverse_observation_indices = np.zeros_like(observation_indices)

        for i in range(len(reverse_observation_indices)):
            reverse_observation_indices[i] = np.where(observation_indices == i)[0][0]
        all_reverse_observation_indices.append(reverse_observation_indices)

    state_dim = 96
    action_dim = 4

    total_sample_dim = 0
    for _ in range(len(policies)):
        total_sample_dim += state_dim + action_dim
    total_sample_dim += state_dim

    num_steps = 256
    
    xt = torch.randn(num_samples, total_sample_dim).to(device)

    sdes, oness = [], []
    sdes_state, ones_state = [], []

    indices_dms = [
        (100*i, 100*i + 196) for i in range(len(policies))
    ]

    indices_sdms = [
        (100*i, 100*i + 96) for i in range(len(policies))
    ]

    for diffusion_model, indices in zip(diffusion_models, indices_dms):
        sde, ones = diffusion_model.configure_sdes(num_steps=num_steps, x_T=xt[:, indices[0]:indices[1]], num_samples=num_samples)
        sdes.append(sde)
        oness.append(ones)

    for diffusion_state_model, indices in zip(diffusion_state_models, indices_sdms):
        sde, ones = diffusion_state_model.configure_sdes(num_steps=num_steps, x_T=xt[:, indices[0]:indices[1]], num_samples=num_samples)
        sdes_state.append(sde)
        ones_state.append(ones)
    
    # x0 = transform_backward(observation*2, all_observation_indices[0]).to(device)

    x0 = (observation*2).to(device)
    x0 = transform_backward(x0, all_reverse_observation_indices[0])

    if target_encoded_obs is not None:
        x1 = (target_encoded_obs*2).to(device)
        x1 = transform_backward(x1, all_reverse_observation_indices[0])

    num_objects = 4
    
    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        pred_primitives, pred_states, epsilons, epsilon_mids, alpha_ts, alpha_tm1s = [], [], [], [], [], []

        for i, sde, ones, sde_state, one_state, indices, indices_state in zip(range(len(sdes)), sdes, oness, sdes_state, ones_state, indices_dms, indices_sdms):
            
            with torch.no_grad():

                current_obs_ind = torch.Tensor(all_observation_indices[i]).unsqueeze(0).repeat(num_samples, 1).to(device)
                sample = xt[:, indices[0]:indices[1]].clone()
                sample[:, :state_dim] = transform_forward(sample[:, :state_dim], all_observation_indices[i])
                sample[:, state_dim+action_dim:] = transform_forward(sample[:, state_dim+action_dim:], all_observation_indices[i])
                epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, current_obs_ind)
                alpha_ts.append(alpha_t)
                alpha_tm1s.append(alpha_tm1)

                epsilon_m, alpha_t_m, alpha_tm1_m = sde_state.sample_epsilon(t * one_state, sample[:, :state_dim], current_obs_ind)

                pred_primitive = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
                pred_state = (sample[:, :state_dim] - torch.sqrt(1 - alpha_t_m)*epsilon_m) / torch.sqrt(alpha_t_m)
                
                # pred_primitive[:, state_dim:state_dim+action_dim] = policies[i].actor.predict(pred_primitive[:, :state_dim])

                # pred_primitive[:, state_dim+action_dim:] = transition_models[i](torch.cat([pred_primitive[:, :state_dim+action_dim], current_obs_ind], dim=1))
                pred_primitive[:, state_dim+action_dim:] = transition_models[i](pred_primitive[:, :state_dim+action_dim])
                
            # if i==0:
            #     pred_primitive[:, state_dim:state_dim+action_dim] = modify_transition_gradient(
            #         pred_primitive[:, :state_dim],
            #         pred_primitive[:, state_dim:state_dim+action_dim],
            #         pred_primitive[:, state_dim+action_dim:],
            #         transition_models[i],
            #         current_obs_ind,
            #     )

            with torch.no_grad():

                pred_primitive = torch.clip(pred_primitive, -1, 1)
                pred_state = torch.clip(pred_state, -1, 1)
        
                pred_primitive[:, :state_dim] = transform_backward(pred_primitive[:, :state_dim], all_reverse_observation_indices[i])
                pred_primitive[:, state_dim+action_dim:] = transform_backward(pred_primitive[:, state_dim+action_dim:], all_reverse_observation_indices[i])
                pred_state = transform_backward(pred_state, all_reverse_observation_indices[i])

                sample[:, :state_dim] = transform_backward(sample[:, :state_dim], all_reverse_observation_indices[i])
                sample[:, state_dim+action_dim:] = transform_backward(sample[:, state_dim+action_dim:], all_reverse_observation_indices[i])

                epsilon = (sample - torch.sqrt(alpha_t)*pred_primitive) / torch.sqrt(1 - alpha_t)
                epsilon_m = (sample[:, :state_dim] - torch.sqrt(alpha_t_m)*pred_state) / torch.sqrt(1 - alpha_t_m)

                epsilons.append(epsilon)
                epsilon_mids.append(epsilon_m)
                pred_primitives.append(pred_primitive)
                pred_states.append(pred_state)

        alpha_t = alpha_ts[0]
        alpha_tm1 = alpha_tm1s[0]

        total_epsilon = torch.zeros(num_samples, total_sample_dim).to(device)
        
        total_epsilon[:, indices_dms[0][0]:indices_dms[0][1]] += epsilons[0]
        if len(epsilons) > 1:
            total_epsilon[:, indices_dms[1][0]:indices_dms[1][1]] += epsilons[1]
            total_epsilon[:, indices_sdms[1][0]:indices_sdms[1][1]] = gamma1*epsilons[0][:, -state_dim:] + (1 - gamma1 - gamma2)*epsilons[1][:, :state_dim] + gamma2*epsilon_mids[1]
        if len(epsilons) > 2:
            total_epsilon[:, indices_dms[2][0]:indices_dms[2][1]] += epsilons[2]
            total_epsilon[:, indices_sdms[2][0]:indices_sdms[2][1]] = gamma1*epsilons[1][:, -state_dim:] + (1 - gamma1 - gamma2)*epsilons[2][:, :state_dim] + gamma2*epsilon_mids[2]
        if len(epsilons) > 3:
            total_epsilon[:, indices_dms[3][0]:indices_dms[3][1]] += epsilons[3]
            total_epsilon[:, indices_sdms[3][0]:indices_sdms[3][1]] = gamma1*epsilons[2][:, -state_dim:] + (1 - gamma1 - gamma2)*epsilons[3][:, :state_dim] + gamma2*epsilon_mids[3]
        if len(epsilons) > 4:
            total_epsilon[:, indices_dms[4][0]:indices_dms[4][1]] += epsilons[4]
            total_epsilon[:, indices_sdms[4][0]:indices_sdms[4][1]] = gamma1*epsilons[3][:, -state_dim:] + (1 - gamma1 - gamma2)*epsilons[4][:, :state_dim] + gamma2*epsilon_mids[4]
        if len(epsilons) > 5:
            total_epsilon[:, indices_dms[5][0]:indices_dms[5][1]] += epsilons[5]
            total_epsilon[:, indices_sdms[5][0]:indices_sdms[5][1]] = gamma1*epsilons[4][:, -state_dim:] + (1 - gamma1 - gamma2)*epsilons[5][:, :state_dim] + gamma2*epsilon_mids[5]
        if len(epsilons) > 6:
            total_epsilon[:, indices_dms[6][0]:indices_dms[6][1]] += epsilons[6]
            total_epsilon[:, indices_sdms[6][0]:indices_sdms[6][1]] = gamma1*epsilons[5][:, -state_dim:] + (1 - gamma1 - gamma2)*epsilons[6][:, :state_dim] + gamma2*epsilon_mids[6]

        # total_epsilon[:, indices_sdms[1][0]+48:indices_sdms[1][0]+60] = epsilons[-1][:, 48:60]
        
        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        # pred_x0[:, -state_dim:] = transform_backward(
        #     modify_gradient_classifier(
        #         transform_forward(pred_x0[:, -state_dim:], all_observation_indices[-1]), 
        #         torch.Tensor(all_observation_indices[-1]).unsqueeze(0).repeat(num_samples, 1).to(device),
        #         score_model_classifiers[-1], 
        #         1),
        #     all_reverse_observation_indices[-1]
        # )

        end_index = 24

        pred_x0[:, :state_dim] = x0
        pred_x0[:, state_dim+action_dim+12:state_dim+action_dim+end_index] = x0[12:end_index]
        pred_x0[:, state_dim+action_dim+12*num_objects:state_dim+action_dim+96] = x0[12*num_objects:96]

        # pred_x0[:, state_dim+action_dim+36:state_dim+action_dim+48] = x0[36:48]
        # pred_x0[:, state_dim+action_dim+48:state_dim+action_dim+60] = x1[48:60]

        if len(policies) > 1:
            pred_x0[:, 2*(state_dim+action_dim)+12:2*(state_dim+action_dim)+end_index] = x0[12:end_index]
            pred_x0[:, 2*(state_dim+action_dim)+12*num_objects:2*(state_dim+action_dim)+96] = x0[12*num_objects:96]
        if len(policies) > 2:
            pred_x0[:, 3*(state_dim+action_dim)+12:3*(state_dim+action_dim)+end_index] = x0[12:end_index]
            pred_x0[:, 3*(state_dim+action_dim)+12*num_objects:3*(state_dim+action_dim)+96] = x0[12*num_objects:96]
        if len(policies) > 3:
            pred_x0[:, 4*(state_dim+action_dim)+12:4*(state_dim+action_dim)+end_index] = x0[12:end_index]
            pred_x0[:, 4*(state_dim+action_dim)+12*num_objects:4*(state_dim+action_dim)+96] = x0[12*num_objects:96]
        if len(policies) > 4:
            pred_x0[:, 5*(state_dim+action_dim)+12:5*(state_dim+action_dim)+end_index] = x0[12:end_index]
            pred_x0[:, 5*(state_dim+action_dim)+12*num_objects:5*(state_dim+action_dim)+96] = x0[12*num_objects:96]
        if len(policies) > 5:
            pred_x0[:, 6*(state_dim+action_dim)+12:6*(state_dim+action_dim)+end_index] = x0[12:end_index]
            pred_x0[:, 6*(state_dim+action_dim)+12*num_objects:6*(state_dim+action_dim)+96] = x0[12*num_objects:96]
        if len(policies) > 6:
            pred_x0[:, 7*(state_dim+action_dim)+12:7*(state_dim+action_dim)+end_index] = x0[12:end_index]
            pred_x0[:, 7*(state_dim+action_dim)+12*num_objects:7*(state_dim+action_dim)+96] = x0[12*num_objects:96]

        pred_x0 = torch.clip(pred_x0, -1, 1)
        
        xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*torch.randn_like(total_epsilon)

    xt = torch.clip(xt, -1, 1)

    final_states = transform_forward(xt[:, -state_dim:].clone(), all_observation_indices[-1])

    scores = score_model_classifiers[-1](
        torch.cat([final_states, torch.Tensor(all_observation_indices[-1]).unsqueeze(0).repeat(num_samples, 1).to(device)], dim=1)
    ).detach().cpu().numpy().squeeze()

    xt = xt.detach().cpu().numpy()

    print("scores:", scores)

    # arrange samples in descending order of scores
    sorted_indices = np.argsort(scores)[::-1][:5]

    xt = xt[sorted_indices]

    state0 = xt[:, :state_dim]*0.5
    action0 = xt[:, state_dim:state_dim+action_dim]
    state1 = xt[:, state_dim+action_dim:state_dim+action_dim+state_dim]*0.5
    if len(policies) > 1:
        action1 = xt[:, state_dim+action_dim+state_dim:state_dim+action_dim+state_dim+action_dim]
        state2 = xt[:, 2*(state_dim+action_dim):2*(state_dim+action_dim)+state_dim]*0.5
    if len(policies) > 2:
        action2 = xt[:, 2*(state_dim+action_dim)+state_dim:3*(state_dim+action_dim)]
        state3 = xt[:, 3*(state_dim+action_dim):3*(state_dim+action_dim)+state_dim]*0.5
    if len(policies) > 3:
        action3 = xt[:, 3*(state_dim+action_dim)+state_dim:4*(state_dim+action_dim)]
        state4 = xt[:, 4*(state_dim+action_dim):4*(state_dim+action_dim)+state_dim]*0.5
    if len(policies) > 4:
        action4 = xt[:, 4*(state_dim+action_dim)+state_dim:5*(state_dim+action_dim)]
        state5 = xt[:, 5*(state_dim+action_dim):5*(state_dim+action_dim)+state_dim]*0.5
    if len(policies) > 5:
        action5 = xt[:, 5*(state_dim+action_dim)+state_dim:6*(state_dim+action_dim)]
        state6 = xt[:, 6*(state_dim+action_dim):6*(state_dim+action_dim)+state_dim]*0.5
    if len(policies) > 6:
        action6 = xt[:, 6*(state_dim+action_dim)+state_dim:7*(state_dim+action_dim)]
        state7 = xt[:, 7*(state_dim+action_dim):7*(state_dim+action_dim)+state_dim]*0.5

    all_actions = [
        action0,
        action1 if len(policies) > 1 else None,
        action2 if len(policies) > 2 else None,
        action3 if len(policies) > 3 else None,
        action4 if len(policies) > 4 else None,
        action5 if len(policies) > 5 else None,
        action6 if len(policies) > 6 else None,
    ]

    all_states = [
        state0,
        state1,
        state2 if len(policies) > 1 else None,
        state3 if len(policies) > 2 else None,
        state4 if len(policies) > 3 else None,
        state5 if len(policies) > 4 else None,
        state6 if len(policies) > 5 else None,
        state7 if len(policies) > 6 else None,
    ]

    print("all_states:", len(all_states), all_states[0].shape)

    return all_actions, all_states

def eval_state_models(
    diffusion_model_state: Diffusion,
    end_diffusion_model_state: Diffusion,
    observation: np.ndarray,
    target_observation: np.ndarray,
    initial_observation_indices: np.ndarray,
    skill_observation_indices: np.ndarray,
    end_observation_indices: np.ndarray,
    num_samples: int = 5,
    device: str = "auto",
):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    x0 = (observation*2).float().to(device)
    x1 = (target_observation*2).float().to(device)

    num_steps = 256
    num_objects = 4

    initial_reverse_observation_indices = np.zeros_like(initial_observation_indices)

    for i in range(len(initial_reverse_observation_indices)):
        initial_reverse_observation_indices[i] = np.where(initial_observation_indices == i)[0][0]

    skill_reverse_observation_indices = np.zeros_like(skill_observation_indices)

    for i in range(len(skill_reverse_observation_indices)):
        skill_reverse_observation_indices[i] = np.where(skill_observation_indices == i)[0][0]

    end_reverse_observation_indices = np.zeros_like(end_observation_indices)

    for i in range(len(end_reverse_observation_indices)):
        end_reverse_observation_indices[i] = np.where(end_observation_indices == i)[0][0]

    xt = torch.randn(num_samples, 196).to(device)

    x0 = transform_backward(x0, initial_reverse_observation_indices)
    x1 = transform_backward(x1, initial_reverse_observation_indices)

    sde, ones = diffusion_model_state.configure_sdes(num_steps=num_steps, x_T=xt, num_samples=num_samples)

    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        with torch.no_grad():

            sample = xt.clone()
            sample[:, :96] = transform_forward(sample[:, :96], skill_observation_indices)
            sample[:, 100:] = transform_forward(sample[:, 100:], skill_observation_indices)

            mod_skill_observation_indices = torch.Tensor(skill_observation_indices).unsqueeze(0).repeat(num_samples, 1).to(device)

            epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, mod_skill_observation_indices)

            pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)

            pred_x0[:, :96] = transform_backward(pred_x0[:, :96], skill_reverse_observation_indices)
            pred_x0[:, 100:] = transform_backward(pred_x0[:, 100:], skill_reverse_observation_indices)

            pred_x0[:, 12*num_objects:96] = x0[12*num_objects:96]
            pred_x0[:, 12:36] = x0[12:36]
            pred_x0[:, 100 + 12*num_objects:100 + 96] = x0[12*num_objects:96]
            pred_x0[:, 100 + 12:100 + 36] = x0[12:36]
            pred_x0[:, 48:60] = x1[48:60]
            # pred_x0[:, :96] = x0

            pred_x0 = torch.clip(pred_x0, -1, 1)

            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*torch.randn_like(epsilon)

    
    xt = torch.clip(xt, -1, 1)

    xt = xt.detach().cpu().numpy()*0.5

    state1 = xt[:, :96]
    state2 = xt[:, 100:]

    return state1, state2


def eval_tamp_diffusion(
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
    timer = timing.Timer()

    seed = np.random.randint(100000) if seed is None else seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    path = pathlib.Path(path)

    if not path.exists():
        path.mkdir(parents=True)

    for f in path.glob("*.gif"):
        f.unlink()

    for f in path.glob("*.png"):
        f.unlink()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # Load environment.
    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)

    primitives = [
        "pick",
        "place",
        "push",
        "pull",
    ]

    policies = {}

    # Load policies.
    for i, policy_checkpoint in enumerate(policy_checkpoints):
        policy_checkpoint = pathlib.Path(policy_checkpoint)
        policy = agents.load(
            checkpoint=policy_checkpoint, env=env, env_kwargs=env_kwargs, device=device
        )

        assert isinstance(policy, agents.RLAgent)
        policy.eval_mode()
        policy.to(device)

        policies[primitives[i]] = policy

    # Load diffusion models.
    diffusion_transition_models = {}
    diffusion_state_models = {}
    transition_models = {}
    score_model_classifiers = {}

    transition_checkpoints = []

    for diffusion_checkpoint in diffusion_checkpoints:
        transition_checkpoint = diffusion_checkpoint.replace("diffusion_models/v5_trans", "/ssdscratch/umishra31/temporal-policies-main/diffusion_models/v3")
        transition_checkpoints.append(pathlib.Path(transition_checkpoint))

    for i, diffusion_checkpoint in enumerate(diffusion_checkpoints):
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

        diffusion_model_transition.load_state_dict(torch.load(diffusion_checkpoint / "diffusion_model_transition.pt"))
        diffusion_model_state.load_state_dict(torch.load(diffusion_checkpoint / "diffusion_model_state.pt"))
        transition_model.load_state_dict(torch.load(transition_checkpoints[i] / "transition_model.pt"))
        score_model_classifier.load_state_dict(torch.load(diffusion_checkpoint / "score_model_classifier.pt"))

        diffusion_model_transition.to(device)
        diffusion_model_state.to(device)
        transition_model.to(device)
        score_model_classifier.to(device)

        diffusion_transition_models[primitives[i]] = diffusion_model_transition
        diffusion_state_models[primitives[i]] = diffusion_model_state
        transition_models[primitives[i]] = transition_model
        score_model_classifiers[primitives[i]] = score_model_classifier

    # # Load pddl.
    # pddl = symbolic.Pddl(pddl_domain, pddl_problem)

    # Run TAMP.
    num_success = 0
    pbar = tqdm.tqdm(
        seed_generator(num_eval, load_path), f"Evaluate {path.name}", dynamic_ncols=True
    )

    all_success = 0
    all_trials = 0

    for idx_iter, (seed, loaded_plan) in enumerate(pbar):
        # Initialize environment.
        all_trials += 1
        observation, info = env.reset() #seed=np.random.randint(100000))
        seed = info["seed"]
        observation_indices = np.array(info["policy_args"]["observation_indices"])
        state = env.get_state()

        target_obj_pose = np.array(
            [ 5.8420539e-01, -2.7297042e-02,  3.4988631e-02,  4.0854869e-05,
        -2.9513856e-05,  2.5810409e-01,  5.9999999e-02,  5.0000001e-02,
         7.0000000e-02,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]
        )

        task_plans = []
        motion_plans = []
        motion_planner_times = []

        # Task planning outer loop.
        # action_skeleton_generator = task_plan(
        #     pddl=pddl, env=env, max_depth=max_depth, timeout=timeout, verbose=verbose
        # )

        # action_skeletons = []

        for action_skeleton in [env.action_skeleton]: # action_skeleton_generator:

            if len(action_skeleton) >= 2:

                # action_skeletons.append(action_skeleton)

                # if len(action_skeletons) < 3:
                #     continue

                # action_skeletons = np.random.permutation(action_skeletons).tolist()
                # action_skeleton = []

                # for a in action_skeletons:
                #     action_skeleton.extend(a)

                for i in range(len(action_skeleton)):
                    print("action_skeleton:", i, str(action_skeleton[i]))

                # target_primitives = [0, 1, 0, 1, 0, 1] # 3, 1, 0]
                target_primitives = [0, 3, 1, 0]
                # target_primitives = [1, 0, 2]

                selected_primitives = [primitives[i] for i in target_primitives] # [action_skeleton[0], action_skeleton[1]]))]

                print("selected_primitives:", selected_primitives)

                mod_skeleton_selection = 4

                action_skeleton = action_skeleton[:mod_skeleton_selection]
                selected_primitives = selected_primitives[:mod_skeleton_selection]

                timer.tic("motion_planner")
                env.set_primitive(action_skeleton[0])

                observation_indices = np.array(action_skeleton[0].get_policy_args()["observation_indices"])
                reverse_observation_indices = np.zeros_like(observation_indices)

                for i in range(len(reverse_observation_indices)):
                    reverse_observation_indices[i] = np.where(observation_indices == i)[0][0]

                encoded_obs = query_observation_vector(policies[primitives[0]], torch.from_numpy(observation).to(device), action_skeleton[0].get_policy_args())
                target_obj_obs = observation.copy()
                target_obj_obs[4] = target_obj_pose
                target_encoded_obs = query_observation_vector(policies[primitives[0]], torch.from_numpy(target_obj_obs).to(device), action_skeleton[0].get_policy_args())

                sanity_check_encoded_obs = transform_backward(encoded_obs*2, reverse_observation_indices).reshape(8,12)
                sanity_check_obs = policies[primitives[0]].encoder.unnormalize(sanity_check_encoded_obs.unsqueeze(0)*0.5).detach().cpu().numpy()[0]
                sanity_check_obs2 = policies[primitives[1]].encoder.unnormalize(sanity_check_encoded_obs.unsqueeze(0)*0.5).detach().cpu().numpy()[0]
                sanity_check_obs3 = policies[primitives[2]].encoder.unnormalize(sanity_check_encoded_obs.unsqueeze(0)*0.5).detach().cpu().numpy()[0]
                sanity_check_obs4 = policies[primitives[3]].encoder.unnormalize(sanity_check_encoded_obs.unsqueeze(0)*0.5).detach().cpu().numpy()[0]

                assert np.linalg.norm(sanity_check_obs - observation) < 1e-3
                assert np.linalg.norm(sanity_check_obs2 - observation) < 1e-3
                assert np.linalg.norm(sanity_check_obs3 - observation) < 1e-3
                assert np.linalg.norm(sanity_check_obs4 - observation) < 1e-3
                
                selected_policies = [policies[primitive] for primitive in selected_primitives]
                selected_diffusion_transition_models = [diffusion_transition_models[primitive] for primitive in selected_primitives]
                selected_diffusion_state_models = [diffusion_state_models[primitive] for primitive in selected_primitives] # + [diffusion_state_models[primitives[target_primitives[-1]]]]
                selected_transition_models = [transition_models[primitive] for primitive in selected_primitives]
                selected_score_model_classifiers = [score_model_classifiers[primitive] for primitive in selected_primitives]

                # all_pred_states1, all_pred_states2 = eval_state_models(
                #     diffusion_model_state=selected_diffusion_transition_models[-1], # selected_diffusion_state_models[-1],
                #     observation=encoded_obs,
                #     target_observation=target_encoded_obs,
                #     initial_observation_indices=observation_indices,
                #     skill_observation_indices=np.array(action_skeleton[-1].get_policy_args()["observation_indices"])
                # )

                # print("all_pred_states:", all_pred_states1.shape)

                # imgs = []

                # for k in range(len(all_pred_states1)):
                #     if all_pred_states1[k] is None:
                #         continue
                #     s = all_pred_states1[k]
                #     s = s.reshape(8,12)
                #     s = selected_policies[-1].encoder.unnormalize(torch.Tensor(s).unsqueeze(0).to(device)).detach().cpu().numpy()[0]

                #     env.set_observation(s)
                #     img1 = env.render()

                #     s = all_pred_states2[k]
                #     s = s.reshape(8,12)
                #     s = selected_policies[-1].encoder.unnormalize(torch.Tensor(s).unsqueeze(0).to(device)).detach().cpu().numpy()[0]

                #     env.set_observation(s)
                #     img2 = env.render()

                #     img = np.concatenate([img1, img2], axis=1)

                #     imgs.append(img)

                # imgs_all = np.concatenate(imgs, axis=0)

                # Image.fromarray(imgs_all).save(path / f"only_state_record_{idx_iter}_{str(action_skeleton[0])}.png")

                # assert False

                all_actions, all_states = plan(
                    policies=selected_policies,
                    diffusion_models=selected_diffusion_transition_models,
                    diffusion_state_models=selected_diffusion_state_models,
                    transition_models=selected_transition_models,
                    score_model_classifiers=selected_score_model_classifiers,
                    action_skeleton=action_skeleton,
                    observation=encoded_obs,
                    observation_indices=observation_indices,
                    target_encoded_obs=target_encoded_obs,
                )

                # obs1 = transform_observation_from_one_policy_to_another(
                #     observation=observation,
                #     policy_from=policies[primitives[0]],
                #     policy_to=policies[primitives[1]],
                #     action_primitive_to=action_skeleton[1],
                #     env=env,
                # )

                # assert False
            
                t_motion_planner = timer.toc("motion_planner")

                for j in tqdm.tqdm(range(all_actions[0].shape[0])):

                    env.reset(seed=seed)

                    env.set_observation(observation)

                    env.record_start()

                    rewards = []
                    all_obs = [observation]

                    for i, primitive in enumerate(action_skeleton):
                        if all_actions[i] is None:
                            continue
                        # env.record_start()

                        env.set_primitive(primitive)
                        print("primitive:", env.get_primitive(), "action:", action_str(env, all_actions[i][j]))
                        # policy_action = query_policy_actor(policies[primitives[i]], torch.from_numpy(observation).to(device), action_skeleton[i].get_policy_args()).detach().cpu().numpy()
                        action = all_actions[i][j]
                        next_obs, r, _, _, _ = env.step(action)
                        rewards.append(r)
                        all_obs.append(next_obs)
                        # env.step(policy_action)

                        # env.record_stop()

                        # env.record_save(path / f"record_{idx_iter}_{str(action_skeleton[0])}_{j}_{i}.gif", reset=True)

                    print("rewards:", rewards)

                    success = np.sum(rewards) == len(rewards)

                    if success:
                        print("success!:", all_obs)

                    env.record_stop()

                    env.record_save(path / f"record_{idx_iter}_{str(action_skeleton[0])}_{j}_{success}.gif", reset=True)

                imgs_all = []

                for j in range(all_states[0].shape[0]):
                    imgs = []
                    for k in range(len(all_states)):
                        if all_states[k] is None:
                            continue
                        s = all_states[k][j]
                        s = s.reshape(8,12)
                        s = selected_policies[-1].encoder.unnormalize(torch.Tensor(s).unsqueeze(0).to(device)).detach().cpu().numpy()[0]

                        env.set_observation(s)
                        img = env.render()

                        imgs.append(img)

                    imgs = np.concatenate(imgs, axis=1)

                    imgs_all.append(imgs)

                imgs_all = np.concatenate(imgs_all, axis=0)

                Image.fromarray(imgs_all).save(path / f"record_{idx_iter}_{str(action_skeleton[0])}_{success}.png")

                all_success += success

            # assert False
        # assert False
        continue

    print("all_success:", all_success, "all_trials:", all_trials, "success rate:", all_success/all_trials)

def main(args: argparse.Namespace) -> None:
    eval_tamp_diffusion(**vars(args))


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
