#!/usr/bin/env python

##############################################################################
# Code derived from "Learning Object Reorientation for Specific-Posed Placement"
# Wada et al. (2022) https://github.com/wkentaro/reorientbot
##############################################################################

import numpy as np
import torch


class ScoreModelMLP(torch.nn.Module):
    def __init__(self, 
            out_channels=128,
            state_dim=96,
            sample_dim=97
        ):

        super().__init__()

        self.state_dim = state_dim

        self.fc_scores = torch.nn.Sequential(
            torch.nn.Linear(state_dim, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, 1)
        )

    def forward(
            self,
            samples, # B x P, P = sample_dim
        ):
        
        scores = self.fc_scores(samples) # B x A x 3

        return torch.sigmoid(scores)

class TransitionModel(torch.nn.Module):
    def __init__(
            self, 
            sample_dim=100,
            state_dim=96,
            out_channels=128
        ):
        super().__init__()

        self.state_dim = state_dim

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(sample_dim-state_dim, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, state_dim),
        )

    def forward(
            self,
            samples # B x P, P = sample_dim
        ):

        output = self.fc_output(samples)
        
        return output