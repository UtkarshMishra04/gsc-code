#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Union
import pickle
import json
import numpy as np
import torch
import tqdm
from PIL import Image
from temporal_policies import agents, envs
from temporal_policies.utils import random, tensors

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

        samples = samples - 0.05 * samples.grad - 0.02 * torch.randn_like(samples)

        samples = samples.detach()

        prev_loss = loss.item()        

    return samples

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
    device: torch.device = "cpu"
) -> np.ndarray:
    """Samples an action from the diffusion model."""
    # Sample action from diffusion model.

    num_steps = 256
    sample_dim = state_dim + action_dim + state_dim

    xt = torch.zeros(num_samples, sample_dim).to(device)

    observation_indices = torch.Tensor(observation_indices).to(device).unsqueeze(0).repeat(num_samples, 1)

    sde, ones = diffusion_model.configure_sdes(num_steps=num_steps, x_T=xt, num_samples=num_samples)

    x0 = torch.Tensor(np.array(obs0)*2).to(device)

    import pickle

    with open("xt_x0.pkl", "wb") as f:
        pickle.dump([xt.cpu().numpy(), x0.cpu().numpy()], f)
    
    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, xt, observation_indices)

        with open(f"check_data/epsilon_{t}.pkl", "wb") as f:
            pickle.dump((
                xt.detach().cpu().numpy(),
                observation_indices.detach().cpu().numpy(),
                epsilon.detach().cpu().numpy(),
                alpha_t.detach().cpu().numpy(),
                alpha_tm1.detach().cpu().numpy()
            ), f
            )
        
        pred = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
        
        pred[:, :state_dim] = x0
        # pred[:, state_dim:state_dim+action_dim] = policy.actor.predict(pred[:, :state_dim]/2)
        if use_transition_model:
            pred[:, state_dim+action_dim:] = transition_model(torch.cat([pred[:, :state_dim+action_dim], observation_indices], dim=1))
        # pred[:, state_dim+action_dim+48:] = x0[48:]
        # pred[:, state_dim+action_dim+6:state_dim+action_dim+12] = x0[6:12]
        # pred[:, state_dim+action_dim+18:state_dim+action_dim+24] = x0[18:24]
        # pred[:, state_dim+action_dim+24:state_dim+action_dim+36] = x0[24:36]
        # pred[:, state_dim+action_dim+42:state_dim+action_dim+48] = x0[42:48]

        pred = torch.clip(pred, -1, 1)

        # print("pred:", pred[0][100:])

        # assert False

        epsilon = (xt - torch.sqrt(alpha_t)*pred) / torch.sqrt(1 - alpha_t)

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)

        assert torch.norm(pred_x0 - pred) < 1e-3

        # pred_x0[:, state_dim+action_dim:] = modify_gradient_classifier(pred_x0[:, state_dim+action_dim:], classifier, 1)

        # pred_x0[:, 12:24] = x0[12:24]
        # pred_x0[:, state_dim+action_dim+12:state_dim+action_dim+24] = x0[12:24]

        pred_x0 = torch.clip(pred_x0, -1, 1)

        new_epsilon = torch.randn_like(epsilon)
        
        xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

        with open(f"check_data/new_epsilon_{t}.pkl", "wb") as f:
            pickle.dump((
                new_epsilon.detach().cpu().numpy()
            ), f
            )

    xt = xt.detach().cpu().numpy()

    initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    print("initial_state:", initial_state.shape, "observation_indices:", observation_indices.shape)

    scores = classifier(torch.cat([torch.Tensor(initial_state).to(device), observation_indices], dim=1)).detach().cpu().numpy().squeeze()

    print("scores:", scores)

    # arrange samples in descending order of scores
    sorted_indices = np.argsort(scores)[::-1]

    xt = xt[sorted_indices]

    import pickle

    with open("result_xt.pkl", "wb") as f:
        pickle.dump(xt, f)

    initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    print("initial_state:", initial_state[0])
    print("action:", action[0])
    # assert False
    print("final_state:", final_state[0])

    # assert False, use_transition_model

    return action, final_state*0.5

def evaluate_episodes(
    env: envs.Env,
    policy: agents.RLAgent,
    diffusion_model_transition: Diffusion,
    diffusion_model_state: Diffusion,
    transition_model: TransitionModel,
    score_model_classifier: ScoreModelMLP,
    obs_preprocessor: Any,
    num_episodes: int,
    path: pathlib.Path,
    verbose: bool,
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

    diffusion_model_transition.to(device)
    diffusion_model_state.to(device)
    transition_model.to(device)
    score_model_classifier.to(device)

    all_results = []

    # remove all gif files from path
    for f in path.iterdir():
        if f.suffix == ".gif" or f.suffix == ".png":
            f.unlink()

    for i in pbar:
        # Evaluate episode.
        observation, reset_info = env.reset() #seed=seed)
        print("reset_info:", reset_info, "env.task", env.task)
        seed = reset_info["seed"]
        initial_observation = observation
        observation_indices = reset_info["policy_args"]["observation_indices"]
        if verbose:
            print("primitive:", env.get_primitive())
            print("reset_info:", reset_info)

        rewards = []
        done = False

        while not done:
            obs0 = obs_preprocessor(observation, reset_info["policy_args"])
            policy_action = query_policy_actor(policy, observation, reset_info["policy_args"])

            actions, pred_states = get_action_from_diffusion(
                policy,
                diffusion_model_transition,
                transition_model,
                score_model_classifier,
                obs0,
                observation_indices=observation_indices,
                use_transition_model=False,
                device=device
            )

            actions_dummy, pred_states_wo_transition = get_action_from_diffusion(
                policy,
                diffusion_model_transition,
                transition_model,
                score_model_classifier,
                obs0,
                observation_indices=observation_indices,
                use_transition_model=True,
                device=device
            )

            if verbose:
                print("observation:", observation_str(env, observation))
                print("observation tensor:", observation)
                # print("action:", action_str(env, action))

            print("actions:", actions)
            print("policy_action:", policy_action)

            for j, action in enumerate(actions):

                env.reset(seed=seed)
                env.set_observation(initial_observation)

                env.record_start()

                try:
                    # action = policy_action
                    observation, reward, terminated, truncated, step_info = env.step(action)
                except Exception as e:
                    continue

                if verbose:
                    print("step_info:", step_info)
                    print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")

                rewards.append(reward)
                done = terminated or truncated

                predicted_observation = transition_model(torch.Tensor(np.concatenate([obs0, action, observation_indices])).unsqueeze(0).to(device)).detach().cpu()
                predicted_score_0 = score_model_classifier(torch.Tensor(np.concatenate([obs0, observation_indices])).unsqueeze(0).to(device)).detach()
                predicted_score_1 = score_model_classifier(torch.cat([predicted_observation.clone(), torch.Tensor(observation_indices).unsqueeze(0)], dim=-1).to(device)).detach().cpu()
                predicted_observation = policy.encoder.decode(predicted_observation.to(device), reset_info["policy_args"]).cpu().numpy()
                diffusion_next_observation = policy.encoder.decode(torch.Tensor(pred_states[j]).unsqueeze(0).to(device), reset_info["policy_args"]).cpu().numpy()
                diffusion_next_observation_wo_transition = policy.encoder.decode(torch.Tensor(pred_states_wo_transition[j]).unsqueeze(0).to(device), reset_info["policy_args"]).cpu().numpy()
                
                # print("observation:", np.max(np.abs(obs_preprocessor(observation, reset_info["policy_args"]) - predicted_observation)))
                # print("decoded_observation:", predicted_observation, predicted_observation.shape)
                print("max error", np.max(np.abs(observation - predicted_observation))/(np.max(np.abs(observation)) - np.min(np.abs(observation))))
                print("predicted_score_0:", predicted_score_0)
                print("predicted_score_1:", predicted_score_1)

                print("rewards:", rewards)
                print("done:", done)
                # print("obs0", obs0)
                # print("obs1", obs_preprocessor(observation, reset_info["policy_args"]))
                # print("pred_states:", pred_states[j])

                success = sum(rewards) > 0.0

                env.record_stop()

                if success:
                    env.record_save(path / f"eval_{i}_{j}_success.gif", reset=True)
                    img = env.render()
                    img_true = np.array(img, dtype=np.uint8)
                    env.set_observation(predicted_observation)
                    img = env.render()
                    img_pred = np.array(img, dtype=np.uint8)
                    env.set_observation(diffusion_next_observation)
                    img = env.render()
                    img_diffusion = np.array(img, dtype=np.uint8)
                    env.set_observation(diffusion_next_observation_wo_transition)
                    img = env.render()
                    img_diffusion_wo_transition = np.array(img, dtype=np.uint8)

                    img_both = np.concatenate([img_true, img_pred, img_diffusion, img_diffusion_wo_transition], axis=1)
                    Image.fromarray(img_both).save(path / f"eval_{i}_{j}_pred.png")
                else:
                    env.record_save(path / f"eval_{i}_{j}_fail.gif", reset=True)

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
    with open(path / f"results_{seed}.json", "w") as f:
        json.dump(
            {
                "num_episodes": num_episodes,
                "num_successes": num_successes,
            }, f)



def evaluate_diffusion(
    checkpoint: Union[str, pathlib.Path],
    diffusion_checkpoint: Union[str, pathlib.Path],
    num_samples: int = 10,
    env_config: Optional[Union[str, pathlib.Path]] = None,
    debug_results: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    num_episodes: int = 1,
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
        print("env_config:", env_config)
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None
        assert False

    # Load policy.
    policy = agents.load(
        checkpoint=checkpoint, env=env, env_kwargs=env_kwargs, device=device
    )

    assert isinstance(policy, agents.RLAgent)
    policy.eval_mode()
    
    if env is None:
        env = policy.env

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

    evaluate_episodes(
        env=env,
        policy=policy,
        diffusion_model_transition=diffusion_model_transition,
        diffusion_model_state=diffusion_model_state,
        transition_model=transition_model,
        score_model_classifier=score_model_classifier,
        obs_preprocessor=observation_preprocessor,
        num_episodes=num_episodes,
        path=diffusion_checkpoint,
        verbose=verbose,
        seed=seed,
    )


def main(args: argparse.Namespace) -> None:
    evaluate_diffusion(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--diffusion-checkpoint", help="Diffusion checkpoint")
    # parser.add_argument("--env-config", help="Env config")
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
