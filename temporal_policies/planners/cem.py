import functools
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from temporal_policies import agents, dynamics, envs, networks
from temporal_policies.planners import base as planners
from temporal_policies.planners import utils
from temporal_policies.utils import spaces, tensors


class CEMPlanner(planners.Planner):
    """Planner using the Improved Cross Entropy Method."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        num_iterations: int = 8,
        num_samples: int = 128,
        num_elites: int = 16,
        standard_deviation: float = 1.0,
        keep_elites_fraction: float = 0.0,
        population_decay: float = 1.0,
        momentum: float = 0.0,
        device: str = "auto",
    ):
        """Constructs the iCEM planner.

        Args:
            policies: Policies used to evaluate trajecotries.
            dynamics: Dynamics model.
            num_iterations: Number of CEM iterations.
            num_samples: Number of samples to generate per CEM iteration.
            num_elites: Number of elites to select from population.
            standard_deviation: Used to sample random actions for the initial
                population. Will be scaled by the action space.
            keep_elites_fraction: Fraction of elites to keep between iterations.
            population_decay: Population decay applied after each iteration.
            momentum: Momentum of distribution updates.
            device: Torch device.
        """
        super().__init__(policies=policies, dynamics=dynamics, device=device)
        self._num_iterations = num_iterations
        self._num_samples = num_samples
        self._num_elites = max(2, min(num_elites, self.num_samples // 2))
        self._standard_deviation = standard_deviation

        # Improved CEM parameters.
        self._num_elites_to_keep = int(keep_elites_fraction * self.num_elites + 0.5)
        self._population_decay = population_decay
        self._momentum = momentum

    @property
    def num_iterations(self) -> int:
        """Number of CEM iterations."""
        return self._num_iterations

    @property
    def num_samples(self) -> int:
        """Number of samples to generate per CEM iteration."""
        return self._num_samples

    @property
    def num_elites(self) -> int:
        """Number of elites to select from population."""
        return self._num_elites

    @property
    def standard_deviation(self) -> float:
        """Unnormalized standard deviation for sampling random actions."""
        return self._standard_deviation

    @property
    def num_elites_to_keep(self) -> int:
        """Number of elites to keep between iterations."""
        return self._num_elites_to_keep

    @property
    def population_decay(self) -> float:
        """Population decay applied after each iteration."""
        return self._population_decay

    @property
    def momentum(self) -> float:
        """Momentum of distribution updates."""
        return self._momentum

    def _compute_initial_distribution(
        self, observation: torch.Tensor, action_skeleton: Sequence[envs.Primitive]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the initial popoulation distribution.

        The mean is generated by randomly rolling out a random trajectory using
        the dynamics model. The standard deviation is scaled according to the
        action space for each action in the skeleton.

        Args:
            observation: Start observation.
            action_skeleton: List of primitives.

        Returns:
            2-tuple (mean, std).
        """
        T = len(action_skeleton)

        # Roll out a trajectory.
        _, actions = self.dynamics.rollout(
            observation, action_skeleton, self.policies
        )
        mean = actions

        # Scale the standard deviations by the action spaces.
        std = spaces.null_tensor(self.dynamics.action_space, (T,))
        for t, primitive in enumerate(action_skeleton):
            a = self.policies[primitive.idx_policy].action_space
            action_range = torch.from_numpy(a.high - a.low)
            std[t] = self.standard_deviation * 0.5 * action_range

        return mean, std.to(self.device)

    def plan(
        self,
        observation: np.ndarray,
        action_skeleton: Sequence[envs.Primitive],
        return_visited_samples: bool = False,
    ) -> planners.PlanningResult:
        """Runs `num_iterations` of CEM.

        Args:
            observation: Environment observation.
            action_skeleton: List of primitives.
            return_visited_samples: Whether to return the samples visited during planning.

        Returns:
            Planning result.
        """
        best_actions: Optional[np.ndarray] = None
        best_states: Optional[np.ndarray] = None
        p_best_success: float = -float("inf")
        best_values: Optional[np.ndarray] = None
        if return_visited_samples:
            visited_actions_list = []
            visited_states_list = []
            p_visited_success_list = []
            visited_values_list = []

        value_fns = [
            self.policies[primitive.idx_policy].critic for primitive in action_skeleton
        ]
        decode_fns = [
            functools.partial(self.dynamics.decode, primitive=primitive)
            for primitive in action_skeleton
        ]

        with torch.no_grad():
            # Prepare action spaces.
            T = len(action_skeleton)
            actions_low = spaces.null_tensor(self.dynamics.action_space, (T,))
            actions_high = actions_low.clone()
            task_dimensionality = 0
            for t, primitive in enumerate(action_skeleton):
                action_space = self.policies[primitive.idx_policy].action_space
                action_shape = action_space.shape[0]
                actions_low[t, :action_shape] = torch.from_numpy(action_space.low)
                actions_high[t, :action_shape] = torch.from_numpy(action_space.high)
                task_dimensionality += action_shape
            actions_low = actions_low.to(self.device)
            actions_high = actions_high.to(self.device)

            # Scale number of samples to task size
            num_samples = self.num_samples * task_dimensionality

            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.dynamics.device)

            # Initialize distribution.
            mean, std = self._compute_initial_distribution(
                t_observation, action_skeleton
            )
            
            elites = torch.empty(
                (0, *mean.shape), dtype=torch.float32, device=self.device
            )

            # Prepare constant agents for rollouts.
            policies = [
                agents.ConstantAgent(
                    action=spaces.null_tensor(
                        self.policies[primitive.idx_policy].action_space,
                        num_samples,
                        device=self.device,
                    ),
                    policy=self.policies[primitive.idx_policy],
                )
                for t, primitive in enumerate(action_skeleton)
            ]

            for idx_iter in range(self.num_iterations):
                # Sample from distribution.
                samples = torch.distributions.Normal(mean, std).sample((num_samples,))
                samples = torch.clip(samples, actions_low, actions_high)

                # Include the best elites from the previous iteration.
                if idx_iter > 0:
                    samples[: self.num_elites_to_keep] = elites[
                        : self.num_elites_to_keep
                    ]

                # Also include the mean.
                samples[self.num_elites_to_keep] = mean

                # Roll out trajectories.
                for t, policy in enumerate(policies):
                    network = policy.actor.network
                    assert isinstance(network, networks.Constant)
                    network.constant = samples[:, t, : policy.action_space.shape[0]]

                print(
                    "t_observation:", t_observation, t_observation.shape,
                    "policies:", policies,
                    "num_samples:", num_samples,
                    "time_index:", True,
                    self.dynamics
                )

                for t, primitive in enumerate(action_skeleton):
                    print("primitive:", primitive)
                    print("primitive.idx_policy:", primitive.idx_policy)
                    print("primitive.policy:", self.policies[primitive.idx_policy])

                states, _ = self.dynamics.rollout(
                    t_observation,
                    action_skeleton,
                    policies,
                    batch_size=num_samples,
                    time_index=True,
                )

                print("states:", states, states.shape)

                assert False

                # Evaluate trajectories.
                p_success, values, _ = utils.evaluate_trajectory(
                    value_fns, decode_fns, states, actions=samples
                )

                # Select the top trajectories.
                idx_elites = p_success.topk(self.num_elites).indices
                elites = samples[idx_elites]
                idx_best = idx_elites[0]

                # Track best action.
                _p_best_success = p_success[idx_best].cpu().numpy()
                if _p_best_success > p_best_success:
                    p_best_success = _p_best_success
                    best_actions = samples[idx_best].cpu().numpy()
                    best_states = states[idx_best].cpu().numpy()
                    best_values = values[idx_best].cpu().numpy()

                # Update distribution.
                mean = self.momentum * mean + (1 - self.momentum) * elites.mean(dim=0)
                std = self.momentum * std + (1 - self.momentum) * elites.std(dim=0)
                std = torch.clip(std, 1e-4)

                # Decay population size.
                num_samples = int(self.population_decay * num_samples + 0.5)
                num_samples = max(num_samples, 2 * self.num_elites)

                # Convert to numpy.
                if return_visited_samples:
                    visited_actions_list.append(samples.cpu().numpy())
                    visited_states_list.append(states.cpu().numpy())
                    p_visited_success_list.append(p_success.cpu().numpy())
                    visited_values_list.append(values.cpu().numpy())

        assert (
            best_actions is not None
            and best_states is not None
            and best_values is not None
        )

        if return_visited_samples:
            visited_actions = np.concatenate(visited_actions_list, axis=0)
            visited_states = np.concatenate(visited_states_list, axis=0)
            p_visited_success = np.concatenate(p_visited_success_list, axis=0)
            visited_values = np.concatenate(visited_values_list, axis=0)
        else:
            visited_actions = None
            visited_states = None
            p_visited_success = None
            visited_values = None

        return planners.PlanningResult(
            actions=best_actions,
            states=best_states,
            p_success=p_best_success,
            values=best_values,
            visited_actions=visited_actions,
            visited_states=visited_states,
            p_visited_success=p_visited_success,
            visited_values=visited_values,
        )
