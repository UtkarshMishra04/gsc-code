#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Union

import numpy as np
import tqdm
from PIL import Image
from temporal_policies import envs


def train(
    env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    use_curriculum: Optional[int] = None,
    num_env_processes: Optional[int] = None,
) -> None:
    
    if seed is not None:
        np.random.seed(seed)

    if env_config is None:
        raise ValueError("env_config must be specified")

    env_factory = envs.EnvFactory(config=env_config)
    env_kwargs: Dict[str, Any] = {}
    env_kwargs["gui"] = False
    if use_curriculum is not None:
        env_kwargs["use_curriculum"] = bool(use_curriculum)
    if num_env_processes is not None:
        env_kwargs["num_processes"] = num_env_processes

    env = env_factory(**env_kwargs)

    print("\n[scripts.train.train_policy] Env config:")
    pprint(env_factory.config)
    print("")

    while True:
        env.reset()
        done = False
        curr_state = env.get_observation()
        print("curr_state.shape", curr_state, env._objects.keys())
        print("object states", env.object_states())
        while not done:
            action = env.action_space.sample()
            img1 = env.render()
            next_state, reward, done1, done2, info = env.step(action)
            img2 = env.render()

            Image.fromarray(img1).save("img1.png")
            Image.fromarray(img2).save("img2.png")
            
            assert False
            env.render()
        env.close()
    
    env.close()


def main(args: argparse.Namespace) -> None:
    train(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--env-config", "-e", required=True, help="Path to env config")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num-env-processes", type=int, help="Number of env processes")

    main(parser.parse_args())
