import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm
import lnn


class LogicConstraintNet():
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None):
        self.is_discrete = True
        self.device = "cpu"

        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        # self._define_input_dims()

        self.action_low = action_low
        self.action_high = action_high

        self.model = lnn.Model()
        self.a_forward = self.model.add_predicates(1, 'a_forward')
        self.a_backward = self.model.add_predicates(1, 'a_backward')
        self.fact_count = 0

        x = lnn.Variable('x')
        self.model['valid'] = lnn.Or(self.a_forward(x), self.a_backward(x))

    # def forward(self, x: th.tensor) -> th.tensor:
    #    raise Exception("forward is not implemented")
        # return self.network(x)

    def cost_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        # add state-action to model with unknown output
        if(acs[0] == 0):
            self.model.add_facts({
                self.a_forward.name: {
                    f'{self.fact_count}': lnn.TRUE
                },
                self.a_backward.name: {
                    f'{self.fact_count}': lnn.FALSE
                },
                'valid': {
                    f'{self.fact_count}': lnn.UNKNOWN
                }
            })
        else:
            self.model.add_facts({
                self.a_forward.name: {
                    f'{self.fact_count}': lnn.FALSE
                },
                self.a_backward.name: {
                    f'{self.fact_count}': lnn.TRUE
                },
                'valid': {
                    f'{self.fact_count}': lnn.UNKNOWN
                }
            })

        # infer the model's output
        self.model.infer(direction=lnn.UPWARD)

        # calculate cost
        cost = 1 - self.model['valid'].get_facts()[-1][-1].item()

        return [cost]

    # def prepare_data(
    #        self,
    #        obs: np.ndarray,
    #        acs: np.ndarray,
    # ) -> th.tensor:
    #
    #    # EDIT: for the moment we do not use the observations, so we should also not normalize them
    #    # obs = self.normalize_obs(
    #    #    obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
    #    acs = self.reshape_actions(acs)
    #    acs = self.clip_actions(acs, self.action_low, self.action_high)
    #
    #    concat = self.select_appropriate_dims(
    #        np.concatenate([obs, acs], axis=-1))
    #
    #    return th.tensor(concat, dtype=th.float32).to(self.device)

    # def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
    #                  clip_obs: Optional[float] = None) -> np.ndarray:
    #    if mean is not None and var is not None:
    #        mean, var = mean[None], var[None]
    #        obs = (obs - mean) / np.sqrt(var + self.eps)
    #    if clip_obs is not None:
    #        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
    #
    #    return obs

    # def reshape_actions(self, acs):
    #    if self.is_discrete:
    #        acs_ = acs.astype(int)
    #        if len(acs.shape) > 1:
    #            acs_ = np.squeeze(acs_, axis=-1)
    #        acs = np.zeros([acs.shape[0], self.acs_dim])
    #        acs[np.arange(acs_.shape[0]), acs_] = 1.
    #
    #    return acs

    # def select_appropriate_dims(self, x: Union[np.ndarray, th.tensor]) -> Union[np.ndarray, th.tensor]:
    #    return x[..., self.select_dim]

    # def clip_actions(self, acs: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
    #    if high is not None and low is not None:
    #        acs = np.clip(acs, low, high)
    #
    #    return acs

    # def _define_input_dims(self) -> None:
    #    self.select_dim = []
    #    if self.obs_select_dim is None:
    #        self.select_dim += [i for i in range(self.obs_dim)]
    #    elif self.obs_select_dim[0] != -1:
    #        self.select_dim += self.obs_select_dim
    #    if self.acs_select_dim is None:
    #        self.select_dim += [i for i in range(self.acs_dim)]
    #    elif self.acs_select_dim[0] != -1:
    #        self.select_dim += self.acs_select_dim
    #    assert len(self.select_dim) > 0, ""
    #
    #    self.input_dims = len(self.select_dim)

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train(
        self,
        iterations: np.ndarray,
        nominal_obs: np.ndarray,
        nominal_acs: np.ndarray,
        episode_lengths: np.ndarray,
        obs_mean: Optional[np.ndarray] = None,
        obs_var: Optional[np.ndarray] = None,
        current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)

        # Save current network predictions if using importance sampling
        if self.importance_sampling:
            with th.no_grad():
                start_preds = self.forward(nominal_data).detach()

        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # Compute IS weights
            if self.importance_sampling:
                with th.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(), current_preds.clone(),
                                                                             episode_lengths)
                # Break if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0])

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # Get batch data
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None]

                # Make predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)

                # Calculate loss
                if self.train_gail_lambda:
                    nominal_loss = self.criterion(
                        nominal_preds, th.zeros(*nominal_preds.size()))
                    expert_loss = self.criterion(
                        expert_preds, th.ones(*expert_preds.size()))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + expert_loss
                else:
                    expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_loss = th.mean(
                        is_batch * th.log(nominal_preds + self.eps))
                    regularizer_loss = self.regularizer_coeff * \
                        (th.mean(1-expert_preds) + th.mean(1-nominal_preds))
                    loss = (-expert_loss + nominal_loss) + regularizer_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        bw_metrics = {"backward/cn_loss": loss.item(),
                      "backward/expert_loss": expert_loss.item(),
                      "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds + self.eps)).item(),
                      "backward/nominal_loss": nominal_loss.item(),
                      "backward/regularizer_loss": regularizer_loss.item(),
                      "backward/is_mean": th.mean(is_weights).detach().item(),
                      "backward/is_max": th.max(is_weights).detach().item(),
                      "backward/is_min": th.min(is_weights).detach().item(),
                      "backward/nominal_preds_max": th.max(nominal_preds).item(),
                      "backward/nominal_preds_min": th.min(nominal_preds).item(),
                      "backward/nominal_preds_mean": th.mean(nominal_preds).item(),
                      "backward/expert_preds_max": th.max(expert_preds).item(),
                      "backward/expert_preds_min": th.min(expert_preds).item(),
                      "backward/expert_preds_mean": th.mean(expert_preds).item(), }
        if self.importance_sampling:
            stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                            "backward/kl_new_old": kl_new_old.item(),
                            "backward/early_stop_itr": early_stop_itr}
            bw_metrics.update(stop_metrics)

        return bw_metrics
