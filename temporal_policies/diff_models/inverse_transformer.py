import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class ObjectEmbedder(nn.Module):
    """
    Embeds object representations into vector representations.
    (N, C, O) -> (N, C, D)
    """
    def __init__(self, input_size, action_size, hidden_size, num_objects):
        super().__init__()

        self.num_objects = num_objects
        
        self.mlp_x1 = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.mlp_x2 = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.object_mlp = nn.Sequential(
            nn.Linear(num_objects, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x1, x2, object_sequence):

        one_hot_object_sequence = F.one_hot(object_sequence.long(), num_classes=self.num_objects).float() # (1, O, O)

        batch_size = x1.shape[0]
        x1 = self.mlp_x1(x1)
        x2 = self.mlp_x2(x2)

        y = self.object_mlp(one_hot_object_sequence)
        
        x1y = torch.cat([x1, y], dim=2)
        x2y = torch.cat([x2, y], dim=2)

        xy = torch.cat([x1y, x2y], dim=1)

        return xy

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU() #approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        # print(f'[ models/temporal ] DiTBlock input shape: {x.shape}')
        # print(f'[ models/temporal ] DiTBlock c shape: {c.shape} and {self.adaLN_modulation(c).chunk(6, dim=1)[0].shape}')
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, input_size, action_size, num_objects):
        super().__init__()
        self.num_objects = num_objects
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear_x2 = nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        ex2 = self.linear_x2(x)
        return ex2


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=12,
        action_size=4,
        num_objects=8,
        hidden_size=128,
        depth=10,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_objects = num_objects
        self.num_heads = num_heads

        self.x_embedder = ObjectEmbedder(input_size, action_size, int(hidden_size/2), num_objects)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_objects*2, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, action_size, action_size, num_objects)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((2*self.num_objects + 1) ** 0.5))
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(2*self.num_objects))
        # print(f'[ models/temporal ] Positional embedding shape: {pos_embed.shape} and copy to {self.pos_embed.data.shape}')
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.constant_(self.final_layer.linear_x2.weight, 0)
        nn.init.constant_(self.final_layer.linear_x2.bias, 0)


    def forward(self, x1, x2, object_sequence):
        """
        Forward pass of DiT.
        x1: (N, C, O) tensor of spatial inputs (images or latent representations of images)
        x2: (N, C, O) tensor of spatial inputs (images or latent representations of images)
        a1: (N, A) tensor of action inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) object sequence labels
        """

        # print(f'[ models/temporal ] x1 shape: {x1.shape}')
        # print(f'[ models/temporal ] x2 shape: {x2.shape}')

        x = self.x_embedder(x1, x2, object_sequence) 
        # print(f'[ models/temporal ] x_embedder output shape: {x.shape}')
        x = x + self.pos_embed                              # (N, C, O) + (1, C, O) -> (N, C, O)
        
        # print(f'[ models/temporal ] t_embedder output shape: {c.shape}')

        # print(f'[ models/temporal ] blocks input shape: {x.shape}')

        for block in self.blocks:
            x = block(x)                                 # (N, T, D)

        # print(f'[ models/temporal ] blocks output shape: {x.shape}')
        
        # print(f'[ models/temporal ] blocks output shape: {x.shape}')

        ex2 = self.final_layer(x)              # (N, T, C) -> (N, C, O), (N, A), (N, C, O)

        # print(f'[ models/temporal ] final_layer output shape: {ex2.shape}')

        return ex2

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, odd_num=True, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size + 1, dtype=np.float32) if odd_num else np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size + 1]) if odd_num else grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def DiT_L_8(**kwargs):
    return DiT(depth=8, num_heads=4, **kwargs)

class InverseDynamicsModel(torch.nn.Module):
    def __init__(
            self, 
            sample_dim=192,
            state_dim=96,
            action_dim=4,
            out_channels=128,
            num_objects=8
        ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.model = DiT_L_8()

    def forward(
            self,
            samples # B x P, P = sample_dim
        ):

        samples1, samples2, object_sequence = samples[:, :self.state_dim], samples[:, self.state_dim:self.state_dim+self.state_dim], samples[:, -self.num_objects:]

        samples1 = samples1.reshape(samples1.shape[0], self.num_objects, -1) # B x P x 1
        samples2 = samples2.reshape(samples2.shape[0], self.num_objects, -1) # B x P x 1

        output = self.model(samples1, samples2, object_sequence)

        output = torch.mean(output, dim=1)

        output = output.reshape(output.shape[0], -1)
        
        return output

if __name__=="__main__":

    object_sequence = np.array([0, 3, 1, 2, 4, 5, 6, 7])

    inv_dyn = InverseDynamicsModel()

    target_transition = torch.randn(10, 4)

    samples_transition = torch.randn(10, 192)

    samples_transition = torch.cat([samples_transition, torch.tensor(object_sequence).unsqueeze(0).repeat(10, 1).float()], dim=1)

    print(inv_dyn(samples_transition).shape)

    optimizer = torch.optim.Adam(list(inv_dyn.parameters()), lr=1e-3)

    for i in range(100):

        optimizer.zero_grad()

        loss_transition = torch.nn.MSELoss()(inv_dyn(samples_transition), target_transition)

        loss = loss_transition

        loss.backward()

        optimizer.step()

        print(loss_transition)
