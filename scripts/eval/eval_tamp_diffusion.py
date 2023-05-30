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

from temporal_policies.diff_models.unet1D import ScoreNet
from temporal_policies.diff_models.classifier import ScoreModelMLP, TransitionModel
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

def modify_gradient_classifier(
    samples: torch.Tensor,
    score_model: ScoreModelMLP,
    target: int,
    num_grad_steps: int = 50
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

        predicted_score = score_model(samples)

        loss = torch.nn.BCELoss()(predicted_score, torch.ones_like(predicted_score)*target)

        loss = loss.mean()

        loss.backward()

        samples = samples - 0.01 * samples.grad - 0.01 * torch.randn_like(samples)

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
    num_samples: int = 10,
    device: str = "auto",
    closed_loop: bool = False,
    gamma1: float = 0.33,
    gamma2: float = 0.33,
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
    
    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        with torch.no_grad():

            pred_primitives, epsilons, epsilon_mids, alpha_ts, alpha_tm1s = [], [], [], [], []

            for i, sde, ones, sde_state, one_state, indices, indices_state in zip(range(len(sdes)), sdes, oness, sdes_state, ones_state, indices_dms, indices_sdms):
                sample = xt[:, indices[0]:indices[1]]
                sample[:, :state_dim] = transform_forward(sample[:, :state_dim], all_observation_indices[i])
                sample[:, state_dim+action_dim:] = transform_forward(sample[:, state_dim+action_dim:], all_observation_indices[i])
                epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample)
                alpha_ts.append(alpha_t)
                alpha_tm1s.append(alpha_tm1)

                epsilon_m, alpha_t_m, alpha_tm1_m = sde_state.sample_epsilon(t * one_state, sample[:, :state_dim])

                pred_primitive = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)

                mod_x0 = transform_forward(x0, all_observation_indices[i])

                if i == 0:
                    pred_primitive[:, :state_dim] = mod_x0
                
                # pred_primitive[:, state_dim:state_dim+action_dim] = policies[i].actor.predict(pred_primitive[:, :state_dim])
                # pred_primitive[:, state_dim:state_dim+action_dim] += torch.randn_like(pred_primitive[:, state_dim:state_dim+action_dim])*0.05
                # if i >= 0:
                # curr_state = transform_forward(prev_state, all_observation_indices[i])
                # pred_primitive[:, :state_dim] = curr_state
                pred_primitive[:, state_dim+action_dim:] = transition_models[i](pred_primitive[:, :state_dim+action_dim])
                # prev_state = pred_primitive[:, -state_dim:]
                pred_primitive = torch.clip(pred_primitive, -1, 1)
        
                # pred_primitive[:, 12:24] = mod_x0[12:24]
                # pred_primitive[:, 48:96] = mod_x0[48:]
                # pred_primitive[:, state_dim+action_dim+12:state_dim+action_dim+24] = mod_x0[12:24]
                # pred_primitive[:, state_dim+action_dim+48:state_dim+action_dim+96] = mod_x0[48:]

                pred_primitive[:, :state_dim] = transform_backward(pred_primitive[:, :state_dim], all_reverse_observation_indices[i])
                pred_primitive[:, state_dim+action_dim:] = transform_backward(pred_primitive[:, state_dim+action_dim:], all_reverse_observation_indices[i])

                pred_state = (sample[:, :state_dim] - torch.sqrt(1 - alpha_t_m)*epsilon_m) / torch.sqrt(alpha_t_m)
                pred_state = torch.clip(pred_state, -1, 1)

                if i == 0:
                    pred_state[:, :state_dim] = mod_x0[:state_dim]
                
                # pred_state[:, 12:24] = mod_x0[12:24]
                # pred_state[:, 48:96] = mod_x0[48:]

                pred_state = transform_backward(pred_state, all_reverse_observation_indices[i])

                sample[:, :state_dim] = transform_backward(sample[:, :state_dim], all_reverse_observation_indices[i])
                sample[:, state_dim+action_dim:] = transform_backward(sample[:, state_dim+action_dim:], all_reverse_observation_indices[i])

                epsilon = (sample - torch.sqrt(alpha_t)*pred_primitive) / torch.sqrt(1 - alpha_t)
                epsilon_m = (sample[:, :state_dim] - torch.sqrt(alpha_t_m)*pred_state) / torch.sqrt(1 - alpha_t_m)

                epsilons.append(epsilon)
                epsilon_mids.append(epsilon_m)
                pred_primitives.append(pred_primitive)

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

            # total_epsilon[:, :state_dim+action_dim+state_dim] = epsilons[0]
            # if len(epsilons) > 1:
            #     total_epsilon[:, state_dim+action_dim+state_dim:2*(state_dim+action_dim)+state_dim] = epsilons[1][:, state_dim:state_dim+action_dim+state_dim]
            # if len(epsilons) > 2:
            #     total_epsilon[:, 2*(state_dim+action_dim)+state_dim:3*(state_dim+action_dim)+state_dim] = epsilons[2][:, state_dim:state_dim+action_dim+state_dim]
            # if len(epsilons) > 3:
            #     total_epsilon[:, 3*(state_dim+action_dim)+state_dim:] = epsilons[3][:, state_dim:state_dim+action_dim+state_dim]

            # total_epsilon[:, :state_dim+action_dim] = epsilons[0][:, :state_dim+action_dim]
            # if len(epsilons) > 1:
            #     total_epsilon[:, state_dim+action_dim:2*(state_dim+action_dim)] = epsilons[1][:, :state_dim+action_dim]
            # if len(epsilons) > 2:
            #     total_epsilon[:, 2*(state_dim+action_dim):3*(state_dim+action_dim)] = epsilons[2][:, :state_dim+action_dim]
            # if len(epsilons) > 3:
            #     total_epsilon[:, 3*(state_dim+action_dim):4*(state_dim+action_dim)+state_dim] = epsilons[3]

            pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        pred_x0[:, -state_dim:] = transform_backward(
            modify_gradient_classifier(
                transform_forward(pred_x0[:, -state_dim:], all_observation_indices[-1]), 
                score_model_classifiers[-1], 
                1),
            all_reverse_observation_indices[-1]
        )

        # mod_x0 = transform_backward(x0, all_reverse_observation_indices[0])
        pred_x0[:, :state_dim] = x0
        # pred_x0[:, state_dim+action_dim+48:state_dim+action_dim+96] = mod_x0[48:]
        pred_x0[:, state_dim+action_dim+12:state_dim+action_dim+24] = x0[12:24]

        if len(policies) > 1:
            # pred_x0[:, 2*(state_dim+action_dim)+48:2*(state_dim+action_dim)+96] = mod_x0[48:]
            pred_x0[:, 2*(state_dim+action_dim)+12:2*(state_dim+action_dim)+24] = x0[12:24]
        if len(policies) > 2:
            # pred_x0[:, 3*(state_dim+action_dim)+48:3*(state_dim+action_dim)+96] = mod_x0[48:]
            pred_x0[:, 3*(state_dim+action_dim)+12:3*(state_dim+action_dim)+24] = x0[12:24]
        if len(policies) > 3:
            # pred_x0[:, 4*(state_dim+action_dim)+48:4*(state_dim+action_dim)+96] = mod_x0[48:]
            pred_x0[:, 4*(state_dim+action_dim)+12:4*(state_dim+action_dim)+24] = x0[12:24]
        if len(policies) > 4:
            # pred_x0[:, 5*(state_dim+action_dim)+48:5*(state_dim+action_dim)+96] = mod_x0[48:]
            pred_x0[:, 5*(state_dim+action_dim)+12:5*(state_dim+action_dim)+24] = x0[12:24]
        if len(policies) > 5:
            # pred_x0[:, 6*(state_dim+action_dim)+48:6*(state_dim+action_dim)+96] = mod_x0[48:]
            pred_x0[:, 6*(state_dim+action_dim)+12:6*(state_dim+action_dim)+24] = x0[12:24]
        if len(policies) > 6:
            # pred_x0[:, 7*(state_dim+action_dim)+48:7*(state_dim+action_dim)+96] = mod_x0[48:]
            pred_x0[:, 7*(state_dim+action_dim)+12:7*(state_dim+action_dim)+24] = x0[12:24]

        pred_x0 = torch.clip(pred_x0, -1, 1)
        
        xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*torch.randn_like(total_epsilon)

    xt = torch.clip(xt, -1, 1)

    scores = score_model_classifiers[-1](xt[:, -state_dim:]).detach().cpu().numpy().squeeze()

    xt = xt.detach().cpu().numpy()

    final_states = transform_forward(xt[:, -state_dim:], all_observation_indices[-1])
    scores = score_model_classifiers[-1](torch.Tensor(final_states).to(device)).detach().cpu().numpy().squeeze()

    print("scores:", scores)

    # arrange samples in descending order of scores
    sorted_indices = np.argsort(scores)[::-1]

    xt = xt[sorted_indices][:1]

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

    # print("state0:", state0.shape)
    # print("action0:", action0.shape)
    # print("state1:", state1.shape)
    # print("action1:", action1.shape)
    # print("state2:", state2.shape)
    # print("action2:", action2.shape)
    # print("state3:", state3.shape)
    # print("action3:", action3.shape)
    # print("state4:", state4.shape)

    # assert False

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

    # initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    # print("initial_state:", initial_state[0])
    # print("action:", action[0])
    # print("final_state:", final_state[0])

    return all_actions, all_states


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

    for i, diffusion_checkpoint in enumerate(diffusion_checkpoints):
        diffusion_checkpoint = pathlib.Path(diffusion_checkpoint)

        score_model_transition = ScoreNet(
            num_samples=num_samples,
            sample_dim=196,
            condition_dim=0
        )

        score_model_state = ScoreNet(
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
        transition_model.load_state_dict(torch.load(diffusion_checkpoint / "transition_model.pt"))
        score_model_classifier.load_state_dict(torch.load(diffusion_checkpoint / "score_model_classifier.pt"))

        diffusion_model_transition.to(device)
        diffusion_model_state.to(device)
        transition_model.to(device)
        score_model_classifier.to(device)

        diffusion_transition_models[primitives[i]] = diffusion_model_transition
        diffusion_state_models[primitives[i]] = diffusion_model_state
        transition_models[primitives[i]] = transition_model
        score_model_classifiers[primitives[i]] = score_model_classifier

    # Load pddl.
    pddl = symbolic.Pddl(pddl_domain, pddl_problem)

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
        state = env.get_state()

        task_plans = []
        motion_plans = []
        motion_planner_times = []

        # Task planning outer loop.
        action_skeleton_generator = task_plan(
            pddl=pddl, env=env, max_depth=max_depth, timeout=timeout, verbose=verbose
        )

        action_skeletons = []

        for action_skeleton in action_skeleton_generator:

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
                selected_diffusion_state_models = [diffusion_state_models[primitive] for primitive in selected_primitives]
                selected_transition_models = [transition_models[primitive] for primitive in selected_primitives]
                selected_score_model_classifiers = [score_model_classifiers[primitive] for primitive in selected_primitives]

                all_actions, all_states = plan(
                    policies=selected_policies,
                    diffusion_models=selected_diffusion_transition_models,
                    diffusion_state_models=selected_diffusion_state_models,
                    transition_models=selected_transition_models,
                    score_model_classifiers=selected_score_model_classifiers,
                    action_skeleton=action_skeleton,
                    observation=encoded_obs,
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

                    for i, primitive in enumerate(action_skeleton):
                        if all_actions[i] is None:
                            continue
                        # env.record_start()

                        env.set_primitive(primitive)
                        print("primitive:", env.get_primitive(), "action:", action_str(env, all_actions[i][j]))
                        # policy_action = query_policy_actor(policies[primitives[i]], torch.from_numpy(observation).to(device), action_skeleton[i].get_policy_args()).detach().cpu().numpy()
                        action = all_actions[i][j]
                        _, r, _, _, _ = env.step(action)
                        rewards.append(r)
                        # env.step(policy_action)

                        # env.record_stop()

                        # env.record_save(path / f"record_{idx_iter}_{str(action_skeleton[0])}_{j}_{i}.gif", reset=True)

                    print("rewards:", rewards)

                    success = np.sum(rewards) >= 2.0

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
    parser.add_argument("--pddl-domain", help="Pddl domain")
    parser.add_argument("--pddl-problem", help="Pddl problem")
    parser.add_argument(
        "--max-depth", type=int, default=4, help="Task planning search depth"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Task planning timeout"
    )
    args = parser.parse_args()

    main(args)
