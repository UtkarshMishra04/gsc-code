#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from temporal_policies.mixed_diffusion.utils.diff_utils import *
from temporal_policies.diff_models.unet1D import ScoreNet

class CombinedDiffusion(Module):

    def __init__(self, add_datasets, add_modules, subtract_datasets, subtract_modules, sample_dims, state_dim):
        super().__init__()

        self.add_datasets = add_datasets
        self.add_modules = add_modules
        self.subtract_datasets = subtract_datasets
        self.subtract_modules = subtract_modules
        self.sample_dims = sample_dims
        self.state_dim = state_dim

        self.grasp_dim = 3
        self.skill_dim = 6

        self.final_sample_len = sum(sample_dims) - sum([state_dim]*len(subtract_modules))

        if len(add_datasets) == 1 and len(subtract_datasets) == 0:
            self.sample = self.sample_one
        else:
            self.sample = self.sample_all

    @torch.no_grad()
    def sample_all(self, condition, num_samples=16, num_steps=256, return_diffusion=False, grad_fn=None, device='cuda'):
        
        initial_state = condition["initial_state"]
        final_state = condition["final_state"]
        skill = condition["skill"]
        skill_object_index = condition["skill_object_index"]

        filter_state = self.add_datasets[0].get_gripper_object_state(initial_state)
        final_filter_state = final_state

        state_dim = self.state_dim
        grasp_dim = self.grasp_dim
        skill_dim = self.skill_dim

        assert filter_state.shape[0] == state_dim and final_state.shape[0] == state_dim

        replace_mask = np.zeros(self.final_sample_len)
        replace_value = np.zeros(self.final_sample_len)

        replace_mask[:self.state_dim] = 1
        replace_value[:self.state_dim] = filter_state

        replace_mask[-self.state_dim:] = 1
        replace_value[-self.state_dim:] = final_filter_state

        replace_mask[self.state_dim+self.grasp_dim+2:self.state_dim+self.grasp_dim+9] = 1
        replace_mask[self.state_dim+self.grasp_dim+11:self.state_dim+self.grasp_dim+30] = 1

        replace_value[self.state_dim+self.grasp_dim+2:self.state_dim+self.grasp_dim+9] = np.array([final_filter_state[2], 0, 0.707, 0, 0.707, 0, 0])
        replace_value[self.state_dim+self.grasp_dim+11:self.state_dim+self.grasp_dim+30] = filter_state[11:30]

        replace_indices = np.where(replace_mask == 1)[0]
        replace_condition_filtered = replace_value[replace_indices]
        replace_condition_filtered = torch.Tensor(replace_condition_filtered).unsqueeze(0).to(device)

        condition_add_modules = []
        mask_add_modules = []

        j_ind = 0

        for i in range(len(self.add_modules)):
            data = {
                'state': initial_state,
                'skill': skill,
                'skill_object_index': skill_object_index,
            }

            condition_add_modules.append(
                self.add_datasets[i].get_conditions(data)
            )

            sde = self.add_modules[i].gensde
            delta = sde.T / num_steps
            self.add_modules[i].gensde.base_sde.dt = delta

        condition_subtract_modules = []

        for i in range(len(self.subtract_modules)):
            data = {
                'state': initial_state,
                'skill': skill,
                'skill_object_index': skill_object_index,
            }

            condition_subtract_modules.append(
                self.subtract_datasets[i].get_conditions(data)
            )

            sde = self.subtract_modules[i].gensde
            delta = sde.T / num_steps
            self.subtract_modules[i].gensde.base_sde.dt = delta

        base_sde = self.add_modules[0].gensde.base_sde
        N = self.add_modules[0].gensde.N

        x_T  = torch.randn([num_samples, self.final_sample_len], device=device)
        ones = torch.ones([num_samples, 1]).to(x_T)/num_steps

        epsilon_cum = torch.zeros([num_samples, self.final_sample_len], device=device)

        initial_state = torch.Tensor(filter_state).unsqueeze(0).to(x_T)
        initial_state = initial_state.repeat(num_samples, 1)

        diffusion = {
            num_steps: x_T.cpu().numpy()
        }

        all_tm1 = []

        prev_epsilon_cum = epsilon_cum.clone()

        for t in range(num_steps, 0, -1):

            # print("epsilon_cum", epsilon_cum[0], self.sample_dims, self.state_dim)

            condition = torch.Tensor(condition_add_modules[0]).unsqueeze(0).to(x_T)
            condition = condition.repeat(num_samples, 1)

            imp_xT = x_T[:, :state_dim+self.grasp_dim+state_dim]
            epsilon, alpha_t, alpha_tm1 = self.add_modules[0].gensde.sample_epsilon(ones * t, imp_xT, condition)
            epsilon_cum[:, :state_dim+self.grasp_dim+state_dim] += epsilon

            # print("step_1: ", epsilon_cum[0] - prev_epsilon_cum[0], epsilon[0], epsilon.shape)
            # prev_epsilon_cum = epsilon_cum.clone()

            condition = torch.Tensor(condition_add_modules[1]).unsqueeze(0).to(x_T)
            condition = condition.repeat(num_samples, 1)

            imp_xT = x_T[:, state_dim+grasp_dim:state_dim+self.grasp_dim+state_dim+self.skill_dim+state_dim]
            epsilon, alpha_t, alpha_tm1 = self.add_modules[1].gensde.sample_epsilon(ones * t, imp_xT, condition)
            epsilon_cum[:, state_dim+grasp_dim:state_dim+self.grasp_dim+state_dim+self.skill_dim+state_dim] += epsilon

            # print("step_2: ", epsilon_cum[0] - prev_epsilon_cum[0], epsilon[0], epsilon.shape)
            # prev_epsilon_cum = epsilon_cum.clone()
            
            condition = torch.Tensor(condition_subtract_modules[0]).unsqueeze(0).to(x_T)
            condition = condition.repeat(num_samples, 1)

            imp_xT = x_T[:, state_dim+grasp_dim:state_dim+self.grasp_dim+state_dim]
            epsilon, alpha_t, alpha_tm1 = self.subtract_modules[0].gensde.sample_epsilon(ones * t, imp_xT, condition)
            epsilon_cum[:, state_dim+grasp_dim:state_dim+self.grasp_dim+state_dim] -= epsilon

            # print("step_3: ", epsilon_cum[0] - prev_epsilon_cum[0], epsilon[0], epsilon.shape)
            # prev_epsilon_cum = epsilon_cum.clone()

            epsilon_cum = torch.clip(epsilon_cum, -1, 1)

            pred_x0 = (x_T - torch.sqrt(1 - alpha_t)*epsilon_cum) / torch.sqrt(alpha_t)
            pred_x0[:, replace_indices] = replace_condition_filtered
            epsilon_cum = (x_T - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)
            epsilon_cum = torch.clip(epsilon_cum, -1, 1)
            # pred_x0 = torch.clip(pred_x0, -1, 1)
            x_t = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*epsilon_cum
        
            diffusion[t-1] = x_t.cpu().numpy()

            x_T = x_t

        print("epsilon_cum", x_T[0], alpha_t[0], alpha_tm1[0], epsilon_cum[0], pred_x0[0])

        if return_diffusion:
            return diffusion[0], diffusion

        return diffusion[0]

    @torch.no_grad()
    def sample_one(self, condition, num_samples=16, num_steps=256, return_diffusion=False, grad_fn=None, device='cuda'):

        initial_state = condition["initial_state"]
        final_state = condition["final_state"]
        skill = condition["skill"]
        skill_object_index = condition["skill_object_index"]

        filter_state = self.add_datasets[0].get_gripper_object_state(initial_state)
        final_filter_state = final_state

        filter_state = self.add_datasets[0].normalize_state_var(filter_state)
        final_filter_state = self.add_datasets[0].normalize_state_var(final_filter_state)

        state_dim = self.state_dim
        grasp_dim = self.grasp_dim
        skill_dim = self.skill_dim

        assert filter_state.shape[0] == state_dim and final_state.shape[0] == state_dim

        replace_mask = np.zeros(self.final_sample_len)
        replace_value = np.zeros(self.final_sample_len)

        replace_mask[:self.state_dim] = 1
        replace_value[:self.state_dim] = filter_state

        replace_mask[self.state_dim+self.grasp_dim+3:self.state_dim+self.grasp_dim+9] = 1
        replace_mask[self.state_dim+self.grasp_dim+11:self.state_dim+self.grasp_dim+30] = 1

        replace_value[self.state_dim+self.grasp_dim+3:self.state_dim+self.grasp_dim+9] = np.array([0, 0.707, 0, 0.707, 0, 0])
        replace_value[self.state_dim+self.grasp_dim+11:self.state_dim+self.grasp_dim+30] = filter_state[11:30]

        replace_indices = np.where(replace_mask == 1)[0]
        replace_condition_filtered = replace_value[replace_indices]
        replace_condition_filtered = torch.Tensor(replace_condition_filtered).unsqueeze(0).to(device)

        condition_add_modules = []
        mask_add_modules = []

        j_ind = 0

        for i in range(len(self.add_modules)):
            data = {
                'state': initial_state,
                'skill': skill,
                'skill_object_index': skill_object_index,
            }

            condition_add_modules.append(
                self.add_datasets[i].get_conditions(data)
            )

            sde = self.add_modules[i].gensde
            delta = sde.T / num_steps
            self.add_modules[i].gensde.base_sde.dt = delta

        condition_subtract_modules = []

        base_sde = self.add_modules[0].gensde.base_sde
        N = self.add_modules[0].gensde.N

        x_T  = torch.randn([num_samples, self.final_sample_len], device=device)
        ones = torch.ones([num_samples, 1]).to(x_T)/num_steps

        epsilon_cum = torch.zeros([num_samples, self.final_sample_len], device=device)

        initial_state = torch.Tensor(filter_state).unsqueeze(0).to(x_T)
        initial_state = initial_state.repeat(num_samples, 1)

        diffusion = {
            num_steps: x_T.cpu().numpy()
        }

        all_tm1 = []

        prev_epsilon_cum = epsilon_cum.clone()

        for t in range(num_steps, 0, -1):

            # print("epsilon_cum", epsilon_cum[0], self.sample_dims, self.state_dim)

            condition = torch.Tensor(condition_add_modules[0]).unsqueeze(0).to(x_T)
            condition = condition.repeat(num_samples, 1)

            imp_xT = x_T[:, :state_dim+self.grasp_dim+state_dim]
            epsilon, alpha_t, alpha_tm1 = self.add_modules[0].gensde.sample_epsilon(ones * t, imp_xT, condition)
            epsilon_cum[:, :state_dim+self.grasp_dim+state_dim] += epsilon

            # epsilon_cum = torch.clip(epsilon_cum, -1, 1)

            pred_x0 = (x_T - torch.sqrt(1 - alpha_t)*epsilon_cum) / torch.sqrt(alpha_t)
            pred_x0[:, replace_indices] = replace_condition_filtered
            pred_x0 = torch.clip(pred_x0, -1/2, 1/2)
            epsilon_cum = (x_T - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)
            # epsilon_cum = torch.clip(epsilon_cum, -1, 1)
            x_t = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*epsilon_cum
        
            diffusion[t-1] = x_t.cpu().numpy()

            x_T = x_t

        print("epsilon_cum", x_T[0], alpha_t[0], alpha_tm1[0], epsilon_cum[0], pred_x0[0])

        if return_diffusion:
            return diffusion[0], diffusion

        return diffusion[0]

    def convert_sample_to_params(self, sample):

        params = []

        state_dim = self.state_dim

        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        for i in range(sample.shape[0]):
            grasp_param = sample[i, state_dim:state_dim+self.grasp_dim]

            if self.final_sample_len > state_dim+self.grasp_dim+self.state_dim:
                skill_param = sample[i, state_dim+self.grasp_dim+state_dim:state_dim+self.grasp_dim+state_dim+self.skill_dim]
            else:
                skill_param = np.zeros(self.skill_dim)

            grasp_param = self.add_datasets[0].unnormalize_grasp_param(grasp_param)
            skill_param = self.add_datasets[0].unnormalize_skill_param(skill_param)

            params.append((
                grasp_param, skill_param
            ))

        return params