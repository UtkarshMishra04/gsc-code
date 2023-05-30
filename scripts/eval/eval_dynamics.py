#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import tqdm
from PIL import Image
from temporal_policies import agents, envs
from temporal_policies.dynamics import DynamicsFactory, Dynamics
from temporal_policies.utils import configs, random, tensors, spaces

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
    dynamics: Optional[Dynamics],
    policies: Optional[Sequence[agents.RLAgent]],
    seed: Optional[int] = None,
    verbose: bool = False,
    debug: bool = False,
    record: bool = False,
) -> List[float]:
    """Evaluates the policy on one episode."""
    observation, reset_info = env.reset(seed=seed)
    if verbose:
        print("primitive:", env.get_primitive())
        print("reset_info:", reset_info)

    if record:
        env.record_start()

    rewards = []
    done = False

    all_observations = [observation]
    all_actions = []

    while not done:
        action = query_policy_actor(policy, observation, reset_info["policy_args"])
        if verbose:
            print("observation:", observation_str(env, observation))
            print("action:", action_str(env, action), action)
            print("policy:", policy.env.get_primitive(), policy.env.get_primitive().get_policy_args(), policy.env.get_primitive().idx_policy)
        if debug:
            input("step?")

        obs = torch.from_numpy(observation).to(policy.device)
        act = torch.from_numpy(action).to(policy.device)

        num_samples = 1

        predicted, _ = dynamics.rollout(
            observation=obs,
            action_skeleton=[policy.env.get_primitive()],
            policies=policies,
            batch_size=num_samples,
            time_index=True,
        )

        predicted = predicted.detach().cpu().numpy()

        observation, reward, terminated, truncated, step_info = env.step(action)

        img_true = env.render()

        img_preds = []

        for j in range(num_samples):
            pred = predicted[j][1]

            env.set_observation(pred)
            img_pred = env.render()
            img_preds.append(img_pred)

        img_pred = np.concatenate(img_preds, axis=0)

        all_imgs = np.concatenate([img_true, img_pred], axis=0)
        Image.fromarray(all_imgs).save("img_{}.png".format(policy.env.name))

        # Image.fromarray(img_true).save("img_true_{}.png".format(policy.env.name))
        # Image.fromarray(img_pred).save("img_pred_{}.png".format(policy.env.name))

        print("pred error:", np.linalg.norm(observation - predicted))
        print("obs error:", observation - obs.detach().cpu().numpy())

        if verbose:
            print("step_info:", step_info)
            print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")

        rewards.append(reward)
        done = terminated or truncated

        all_observations.append(observation)
        all_actions.append(action)

    if record:
        env.record_stop()

    if debug:
        input("finish?")

    return rewards, {
        "observations": all_observations,
        "actions": all_actions,
    }


def eval(
    path: Union[str, pathlib.Path],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
    gui: Optional[int] = None,
    num_train_steps: Optional[int] = None,
) -> None:

    if seed is not None:
        random.seed(seed)

    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    dynamics_factory = DynamicsFactory(
        checkpoint=dynamics_checkpoint,
        policy_checkpoints=policy_checkpoints,
        env_kwargs=env_kwargs,
        device=device,
    )
    dynamics = dynamics_factory()
    dynamics.plan_mode()

    print("\n[scripts.train.train_dynamics] Dynamics config:")
    pprint(dynamics_factory.config)
    print("\n[scripts.train.train_dynamics] Policy checkpoints:")
    pprint(policy_checkpoints)
    print("")

    policies = []

    print("dynamics.policies:", dynamics.policies)

    # for i, policy in enumerate(dynamics.policies):
    #     policy.env.reset()
    #     primitive = policy.env.get_primitive()
    #     # primitive.set_idx_policy(i)
    #     policy.env.set_primitive(primitive)
    #     print("policy:", policy.env.get_primitive(), policy.env.get_primitive().get_policy_args(), policy.env.get_primitive().idx_policy)

    for i, checkpoint in enumerate(policy_checkpoints):

        # Load env.
        env_kwargs: Dict[str, Any] = {}
        env_kwargs["gui"] = False

        env_config = pathlib.Path(checkpoint).parent / "eval/env_config.yaml"
        try:
            env = envs.load(env_config, **env_kwargs)
        except FileNotFoundError:
            # Default to train env.
            env = None

        if env is not None:
            env.reset()
            primitive = env.get_primitive()
            primitive.set_idx_policy(i)
            env.set_primitive(primitive)

        # Load policy.
        policy = agents.load(
            checkpoint=checkpoint, env=env, env_kwargs=env_kwargs, device=device
        )
        assert isinstance(policy, agents.RLAgent)
        policy.eval_mode()

        policies.append(policy)

    # for policy in policies:
    #     print("policy:", policy.env.get_primitive(), policy.env.get_primitive().get_policy_args(), policy.env.get_primitive().idx_policy)

    for i, checkpoint in enumerate(policy_checkpoints):

        # Load env.
        env_kwargs: Dict[str, Any] = {}
        env_kwargs["gui"] = False

        env_config = pathlib.Path(checkpoint).parent / "eval/env_config.yaml"
        try:
            env = envs.load(env_config, **env_kwargs)
        except FileNotFoundError:
            # Default to train env.
            env = None

        # Load policy.
        policy = policies[i]

        if env is None:
            env = policy.env

        # Load reset seed.
        rewards, obs_act = evaluate_episode(
            policy, env, dynamics, policies, seed=seed, verbose=True, debug=False, record=False
        )

        print("rewards:", rewards)

def main(args: argparse.Namespace) -> None:
    eval(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dynamics-checkpoint',
        '-d',
        type=str,
        required=True,
        help='Path to dynamics checkpoint',
    )
    parser.add_argument(
        "--policy-checkpoints",
        nargs="+",
        type=str,
        help="Path to policy checkpoints",
    )
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--num-train-steps", type=int, help="Number of steps to train")

    main(parser.parse_args())
