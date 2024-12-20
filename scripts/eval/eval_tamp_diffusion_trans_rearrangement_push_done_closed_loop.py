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

import time

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
    num_steps: int = 256,
    gamma: Sequence[float] = [1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 1.0],
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

    num_steps = num_steps
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

    for i in range(len(policies)):
        obs_ind = torch.Tensor(all_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        reverse_obs_ind = torch.Tensor(all_reverse_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        sde, ones = diffusion_models[i].configure_sdes(num_steps=num_steps, x_T=xt[:, indices_dms[i][0]:indices_dms[i][1]], num_samples=num_samples)
        all_sdes.append(sde)
        all_ones.append(ones)
        all_obs_ind.append(obs_ind)
        all_reverse_obs_ind.append(reverse_obs_ind)

    full_next_state_sequence = []

    target_state_indices = (indices_dms[0][1] - state_dim, indices_dms[0][1])

    full_next_state_sequence.append(xt[:, target_state_indices[0]:target_state_indices[1]].clone().detach().cpu().numpy())

    start_time = time.time()

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
            
            with torch.no_grad():
                if use_transition_model: # or i % 2 == 0:
                    pred_x0[:, state_dim+action_dim:] = transition_model(torch.cat([pred_x0[:, :state_dim+action_dim], obs_ind], dim=1))

                # pred_x0 = torch.clip(pred_x0, -1, 1)

                epsilon = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

                epsilon[:, :state_dim] = transform_backward(epsilon[:, :state_dim], reverse_observation_indices)
                epsilon[:, -state_dim:] = transform_backward(epsilon[:, -state_dim:], reverse_observation_indices)

                total_epsilon[:, indices_dm[0]:indices_dm[1]] += epsilon

                all_epsilons.append(epsilon)

                if i > 0:
                    total_epsilon[:, indices_sdm[0]:indices_sdm[1]] = gamma[i]*all_epsilons[i-1][:, -state_dim:] + (1-gamma[i])*all_epsilons[i][:, :state_dim]

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        pred_x0[:, :state_dim] = mod_x0[:state_dim]

        for i in range(len(indices_sdms)):
            pred_x0[:, indices_sdms[i][0]+12:indices_sdms[i][0]+end_index] = mod_x0[12:end_index]
            pred_x0[:, indices_sdms[i][0]+12*num_objects:indices_sdms[i][0]+state_dim] = mod_x0[12*num_objects:state_dim]

            # if i < 5:
            #     pred_x0[:, indices_sdms[i][0]+36:indices_sdms[i][0]+48] = mod_x0[36:48]

        pred_x0[:, -state_dim+12:-state_dim+end_index] = mod_x0[12:end_index]
        pred_x0[:, -state_dim+12*num_objects:] = mod_x0[12*num_objects:]

        # for i in range(len(indices_dms)):
        #     state_1 = pred_x0[:, indices_dms[i][0]:indices_dms[i][0]+state_dim].clone()
        #     state_2 = pred_x0[:, indices_dms[i][1]-state_dim:indices_dms[i][1]].clone()

        #     state_1 = transform_forward(state_1, all_observation_indices[i])
        #     state_2 = transform_forward(state_2, all_observation_indices[i])

        #     state_2[:, 36:] = state_1[:, 36:]

        #     state_1 = transform_backward(state_1, all_reverse_observation_indices[i])
        #     state_2 = transform_backward(state_2, all_reverse_observation_indices[i])

        #     pred_x0[:, indices_dms[i][0]:indices_dms[i][0]+state_dim] = state_1
        #     pred_x0[:, indices_dms[i][1]-state_dim:indices_dms[i][1]] = state_2
            
        with torch.no_grad():

            pred_x0 = torch.clip(pred_x0, -1, 1)

            new_epsilon = torch.randn_like(total_epsilon)

            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

            full_next_state_sequence.append(xt[:, target_state_indices[0]:target_state_indices[1]].clone().detach().cpu().numpy())

    end_time = time.time()

    try:
        action = xt[:, -3*(state_dim+action_dim):-2*(state_dim+action_dim)-state_dim].clone()

        action[:, 0] = -action[:, 0]
        action[:, 1] = -action[:, 1]

        action = torch.clip(action, -1, 1)

        xt[:, -3*(state_dim+action_dim):-2*(state_dim+action_dim)-state_dim] = action
    except:
        pass

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

    selected_next_state_sequence = [
        full_next_state_sequence[i][sorted_indices[0]] for i in range(len(full_next_state_sequence))
    ]

    xt = xt[sorted_indices]

    all_states = []
    all_actions = []

    for i in range(len(policies)):
        all_states.append(xt[:, indices_sdms[i][0]:indices_sdms[i][1]]*0.5)
        all_actions.append(xt[:, indices_sdms[i][1]:indices_sdms[i][1]+action_dim])
    
    all_states.append(xt[:, -state_dim:]*0.5)

    return all_actions, all_states, selected_next_state_sequence, end_time - start_time

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

    given_skill_sequence = target_skill_sequence.copy()

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
        if f.suffix == ".gif" or f.suffix == ".png" or f.suffix == ".pkl":
            f.unlink()

    all_rewards = None

    it = 0

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

        i = 0
        target_skill_sequence = given_skill_sequence.copy()
        target_length = len(target_skill_sequence)

        # env.record_start()

        images = []

        # images.append(env.render())

        all_successive_states = []

        # import pickle

        # with open(path / f"eval_{ep}_7_success.pkl", "rb") as f:
        #     all_successive_states, initial_observation = pickle.load(f)

        # env.set_observation(initial_observation)

        # print("all_successive_states:", len(all_successive_states), len(all_successive_states[0]), all_successive_states[0][0].shape)

        # for i in tqdm.tqdm(range(len(all_successive_states))):

        #     folder_name = "step_" + str(i)

        #     for j in tqdm.tqdm(range(len(all_successive_states[i]))):

        #         file_name = "state_" + str(j) + ".png"

        #         state = all_successive_states[i][j]

        #         curr_state = state.reshape(8, 12)
        #         curr_state = policy.encoder.unnormalize(torch.Tensor(curr_state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        #         env.set_observation(curr_state)
        #         img = Image.fromarray(env.render())

        #         if not (path / folder_name).exists():
        #             (path / folder_name).mkdir(parents=True)

        #         img.save(path / folder_name / file_name)

        # assert False

        all_planner_times = []

        while not done:

            target_skill_sequence = target_skill_sequence[-target_length:]

            print("target_skill_sequence:", target_skill_sequence)

            if len(target_skill_sequence) == 0:
                done = True
                break

            env.set_primitive(env.action_skeleton[i])

            observation = env.get_observation()

            obs0 = obs_preprocessor(observation, env.action_skeleton[i].get_policy_args())
            policy_action = query_policy_actor(policy, observation, env.action_skeleton[i].get_policy_args())

            actions, pred_states, selected_next_state_sequence, t_planner = get_action_from_multi_diffusion(
                policies=[policies[skills[i]] for i in target_skill_sequence],
                diffusion_models=[diffusion_models[skills[i]] for i in target_skill_sequence],
                transition_models=[transition_models[skills[i]] for i in target_skill_sequence],
                classifiers=[classifier_models[skills[i]] for i in target_skill_sequence],
                obs0=obs0,
                action_skeleton=env.action_skeleton[i:],
                use_transition_model=False,
                num_objects=5,
                end_index=36,
                device=device,
                num_steps=128,
            )

            all_planner_times.append(t_planner)

            all_successive_states.append(selected_next_state_sequence)

            current_observation = env.get_observation()
            current_state = env.get_state()

            for a in range(actions[0].shape[0]):
                action = actions[0][a]

                env.set_observation(current_observation)
                env.set_state(current_state)

                try:
                    observation, reward, terminated, truncated, step_info = env.step(action)
                except Exception as e:
                    done = True
                    assert False, e

                if reward > 0:
                    # images.append(env.render())
                    break

            print("Executing action:", env.action_skeleton[i], "reward:", reward, "terminated:", terminated, "truncated:", truncated, "step_info:", step_info)

            i += 1
            target_length -= 1

            rewards.append(reward)
            print("rewards:", rewards, len(rewards), len(env.action_skeleton))

            if reward == 0 or len(rewards) == len(env.action_skeleton):
                done = True

        print("all_planner_times:", all_planner_times)

        success = rewards[-1] > 0

        # env.record_stop()

        # imgs = np.concatenate(images, axis=1)

        # if success:
        #     env.record_save(path / f"eval_{ep}_{i}_success.gif", reset=True)
        #     Image.fromarray(imgs).save(path / f"eval_{ep}_{i}_success.png") 
        #     import pickle

        #     with open(path / f"eval_{ep}_{i}_success.pkl", "wb") as f:
        #         pickle.dump((all_successive_states, initial_observation), f)
        # else:
        #     env.record_save(path / f"eval_{ep}_{i}_fail.gif", reset=True)
        #     Image.fromarray(imgs).save(path / f"eval_{ep}_{i}_fail.png") 

        if all_rewards is None:
            all_rewards = np.array(rewards)
        else:
            shape1 = all_rewards.shape[0]
            rewards = np.array(rewards)
            shape2 = rewards.shape[0]
            if shape1 > shape2:
                rewards = np.pad(rewards, (0, shape1 - shape2), "constant", constant_values=0)
            all_rewards += rewards

        num_successes += success
        pbar.set_postfix(
            {"rewards": all_rewards, "successes": f"{num_successes} / {num_episodes}"}
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

    target_skill_sequence = [4, 2, 1, 0, 1, 4, 3] #1, 0, 1, 0, 3]
    target_length = len(target_skill_sequence)
    num_episodes = num_eval

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