#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Union
import pickle
import numpy as np
import torch
import tqdm

from temporal_policies import agents, envs
from temporal_policies.utils import random, tensors


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


def evaluate_episode(
    policy: agents.RLAgent,
    env: envs.Env,
    seed: Optional[int] = None,
    verbose: bool = False,
    debug: bool = False,
    record: bool = False,
) -> List[float]:
    """Evaluates the policy on one episode."""
    observation, reset_info = env.reset() #seed=seed)
    if verbose:
        print("primitive:", env.get_primitive())
        print("reset_info:", reset_info)

    if record:
        env.record_start()

    reset_observation = observation
    base_action = query_policy_actor(policy, observation, reset_info["policy_args"])

    all_save_dicts_pos = []
    all_save_dicts_neg = []

    for _ in range(100):

        env.set_observation(reset_observation)

        rewards = []
        done = False

        all_observations = [observation]
        all_actions = []
        all_reset_info = [reset_info]

        while not done:

            action = base_action + np.random.normal(0, 0.3, size=base_action.shape)
            action = np.clip(action, -1.0, 1.0)
            
            if verbose:
                print("observation:", observation_str(env, observation))
                print("observation tensor:", observation)
                print("action:", action_str(env, action))
            if debug:
                input("step?")

            observation, reward, terminated, truncated, step_info = env.step(action)
            if verbose:
                print("step_info:", step_info)
                print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")

            rewards.append(reward)
            done = terminated or truncated

            all_observations.append(observation)
            all_actions.append(action)

        if sum(rewards) > 0.0:
            all_save_dicts_pos.append({
                "observations": all_observations,
                "actions": all_actions,
                "reset_info": all_reset_info,
                "rewards": rewards,
            })
        else:
            all_save_dicts_neg.append({
                "observations": all_observations,
                "actions": all_actions,
                "reset_info": all_reset_info,
                "rewards": rewards,
            })

    if record:
        env.record_stop()

    if debug:
        input("finish?")

    return all_save_dicts_pos, all_save_dicts_neg


def evaluate_episodes(
    env: envs.Env,
    policy: agents.RLAgent,
    num_episodes: int,
    path: pathlib.Path,
    verbose: bool,
) -> None:
    """Evaluates policy for the given number of episodes."""
    num_successes = 0
    pbar = tqdm.tqdm(
        range(num_episodes),
        desc=f"Evaluate {env.name}",
        dynamic_ncols=True,
    )

    all_results = []
    all_neg_results = []

    for i in pbar:
        # Evaluate episode.
        all_save_dicts_pos, all_save_dicts_neg = evaluate_episode(
            policy, env, verbose=verbose, debug=False, record=False # True
        )
        num_successes += len(all_save_dicts_pos)
        if len(all_save_dicts_pos) > 0:
            pbar.set_postfix(
                {"rewards": all_save_dicts_pos[0]["rewards"], "successes": f"{num_successes} / {num_episodes}"}
            )
        elif len(all_save_dicts_neg) > 0:
            pbar.set_postfix(
                {"rewards": all_save_dicts_neg[0]["rewards"], "successes": f"{num_successes} / {num_episodes}"}
            )
        else:
            pbar.set_postfix(
                {"rewards": "None", "successes": f"{num_successes} / {num_episodes}"}
            )

        all_results.extend(all_save_dicts_pos)
        all_neg_results.extend(all_save_dicts_neg)

        # Save recording.
        # suffix = "" if success else "_fail"
        # env.record_save(path / env.name / f"eval_{i}{suffix}.gif", reset=True)

        # if isinstance(env, envs.pybullet.TableEnv):
        #     # Save reset seed.
        #     with open(path / env.name / f"results_{i}.npz", "wb") as f:
        #         save_dict = {
        #             "seed": env._seed,
        #         }
        #         np.savez_compressed(f, **save_dict)  # type: ignore

        if len(all_results) > 0 and (i + 1) % 10 == 0:
            with open(f"obs_act_data_{env.name}_hook_eval_random.pkl", "wb") as f:
                pickle.dump(all_results, f)

            with open(f"obs_act_data_{env.name}_hook_eval_random_neg.pkl", "wb") as f:
                pickle.dump(all_neg_results, f)

def evaluate_policy(
    checkpoint: Union[str, pathlib.Path],
    env_config: Optional[Union[str, pathlib.Path]] = None,
    debug_results: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    num_episodes: int = 100,
    seed: Optional[int] = None,
    gui: Optional[bool] = None,
    verbose: bool = True,
    device: str = "auto",
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
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None

    # Load policy.
    policy = agents.load(
        checkpoint=checkpoint, env=env, env_kwargs=env_kwargs, device=device
    )

    assert isinstance(policy, agents.RLAgent)
    policy.eval_mode()

    if env is None:
        env = policy.env

    if debug_results is not None:
        # Load reset seed.
        with open(debug_results, "rb") as f:
            seed = int(np.load(f, allow_pickle=True)["seed"])
        evaluate_episode(
            policy, env, seed=seed, verbose=verbose, debug=True, record=False
        )
    elif path is not None:
        # Evaluate episodes.
        evaluate_episodes(env, policy, num_episodes, pathlib.Path(path), verbose)


def main(args: argparse.Namespace) -> None:
    evaluate_policy(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--debug-results", type=str, help="Path to results_i.npz file.")
    parser.add_argument("--path", help="Path for output plots")
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--device", default="auto", help="Torch device")
    args = parser.parse_args()

    main(args)
