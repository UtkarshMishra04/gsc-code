import argparse
import datetime
import pathlib
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import tqdm

from temporal_policies import envs, planners
from temporal_policies.envs.pybullet.table import primitives as table_primitives
from temporal_policies.utils import timing
from temporal_policies.envs.pybullet import real


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


def evaluate_realworld(
    config: Union[str, pathlib.Path],
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    closed_loop: int,
    load_npz: Optional[Union[str, pathlib.Path]] = None,
    gui: Optional[int] = None,
) -> None:
    timer = timing.Timer()

    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)
    assert isinstance(env, envs.pybullet.TableEnv)
    if not isinstance(env.robot.arm, real.arm.Arm):
        raise RuntimeError("Change env config to use real arm.")

    planner = planners.load(
        config=config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        scod_checkpoints=scod_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    path = pathlib.Path(path) / pathlib.Path(config).stem
    path.mkdir(parents=True, exist_ok=True)

    if load_npz is not None:
        with open(load_npz, "rb") as f:
            npz = np.load(f, allow_pickle=True)
            plan = planners.PlanningResult(
                actions=np.array(npz["actions"]),
                states=np.array(npz["states"]),
                p_success=npz["p_success"].item(),
                values=np.array(npz["values"]),
            )

    num_success = 0
    pbar = tqdm.tqdm(range(num_eval), f"Evaluate {path.name}", dynamic_ncols=True)
    for idx_iter in pbar:
        observation, info = env.reset()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if load_npz is not None:
            raise NotImplementedError
        else:
            if closed_loop:
                planning_fn = planners.run_closed_loop_planning
            else:
                planning_fn = planners.run_open_loop_planning
            rewards, plan, t_planner = planning_fn(
                env,
                env.action_skeleton,
                planner,
                timer=timer,
                gif_path=path / f"realworld_{timestamp}.gif",
            )

        if rewards.prod() > 0:
            num_success += 1
        pbar.set_postfix(
            dict(
                success=rewards.prod(),
                **{f"r{t}": r for t, r in enumerate(rewards)},
                num_successes=f"{num_success} / {num_eval}",
            )
        )
        print("success:", rewards.prod(), rewards)
        print("predicted success:", plan.p_success, plan.values)
        if closed_loop:
            print(
                "visited predicted success:",
                plan.p_visited_success,
                plan.visited_values,
            )
        for primitive, action in zip(env.action_skeleton, plan.actions):
            if isinstance(primitive, table_primitives.Primitive):
                primitive_action = str(primitive.Action(action))
                primitive_action = primitive_action.replace("\n", "\n  ")
                print("-", primitive, primitive_action[primitive_action.find("{") :])
            else:
                print("-", primitive, action)
        print("time:", t_planner)

        with open(path / f"results_{idx_iter}.npz", "wb") as f:
            save_dict = {
                "args": {
                    "config": config,
                    "env_config": env_config,
                    "policy_checkpoints": policy_checkpoints,
                    "dynamics_checkpoint": dynamics_checkpoint,
                    "device": device,
                    "num_eval": num_eval,
                    "path": path,
                },
                "observation": observation,
                "actions": plan.actions,
                "states": plan.states,
                "scaled_actions": scale_actions(plan.actions, env, env.action_skeleton),
                "p_success": plan.p_success,
                "values": plan.values,
                "rewards": rewards,
                # "visited_actions": plan.visited_actions,
                # "scaled_visited_actions": scale_actions(
                #     plan.visited_actions, env, action_skeleton
                # ),
                # "visited_states": plan.visited_states,
                "p_visited_success": plan.p_visited_success,
                # "visited_values": plan.visited_values,
                "t_planner": t_planner,
                "action_skeleton": list(map(str, env.action_skeleton)),
            }
            np.savez_compressed(f, **save_dict)  # type: ignore

    print("Successes:", num_success, "/", num_eval)


def main(args: argparse.Namespace) -> None:
    evaluate_realworld(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "--planner-config", "--planner", "-c", help="Path to planner config"
    )
    parser.add_argument("--env-config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument("--scod-checkpoints", "-s", nargs="+", help="SCOD checkpoints")
    parser.add_argument("--dynamics-checkpoint", "-d", help="Dynamics checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=10000, help="Number of eval iterations"
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument("--load-npz", help="Load already generated planning results")
    parser.add_argument(
        "--closed-loop", default=1, type=int, help="Run closed-loop planning"
    )
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    args = parser.parse_args()

    main(args)
