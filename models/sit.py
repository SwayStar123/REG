# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from functools import lru_cache
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models.memory import ConceptBank, compute_concept_losses


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim, use_mlp: bool = False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_mlp:
            # self.linear = nn.Sequential(
            #     nn.Linear(hidden_size, hidden_size),
            #     nn.GELU(),
            #     nn.Linear(hidden_size, patch_size * patch_size * out_channels),
            # )
            self.linear_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, cls_token_dim),
            )
        else:
            # self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            self.linear_cls = nn.Linear(hidden_size, cls_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=True)
        )

    def forward(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        if cls is None:
            x = self.linear(x)
            return x, None
        else:
            cls_token = self.linear_cls(x[:, 0]).unsqueeze(1)
            # x = self.linear(x[:, 1:])
            return x, cls_token.squeeze(1)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels+max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        return dct


    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2 ** 0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs

class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        cls_token_dim=768,
        uvit_skips=True,
        subpatch_size=4,
        num_registers=256,
        n_concepts=4096,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.uvit_skips = uvit_skips
        self.subpatch_size = subpatch_size

        # Input embedders
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.subpatch_embedder = NerfEmbedder((subpatch_size**2)*in_channels, hidden_size, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.cls_projectors2 = nn.Linear(in_features=cls_token_dim, out_features=hidden_size, bias=True)
        self.wg_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        
        # Skip connection linear layers for second half of blocks
        # Each layer projects concatenated (skip + activation) back to hidden_size
        if uvit_skips:
            half_depth = depth // 2
            self.skip_linears = nn.ModuleList([
                nn.Linear(2 * hidden_size, hidden_size, bias=True) for _ in range(half_depth)
            ])
        
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])

        z_dim = self.z_dims[0]
        self.cls_token_dim = z_dim

        self.pixel_calculators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size+hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(4)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.cls_token_dim)
        self.nerf_final_layer = NerfFinalLayer(hidden_size, (subpatch_size**2)*self.out_channels)

        self.registers = nn.Parameter(torch.randn(num_registers, hidden_size))
        self.concept_banks = nn.ModuleList([
            ConceptBank(n_concepts, hidden_size, key_dim=256, n_registers=num_registers, topk_per_register=1) for _ in range(depth-1)
        ])
        self.n_concepts = n_concepts

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
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=1, extra_tokens=1
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize skip connection linear layers:
        if self.uvit_skips:
            for skip_linear in self.skip_linears:
                nn.init.xavier_uniform_(skip_linear.weight)
                nn.init.constant_(skip_linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.nerf_final_layer.linear.weight, 0)
        nn.init.constant_(self.nerf_final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.linear_cls.weight, 0)
        nn.init.constant_(self.final_layer.linear_cls.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, return_logvar=False, cls_token=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        #cat with cls_token
        N, C, H, W = x.shape
        P = self.patch_size
        SP = self.subpatch_size

        patches = x.view(N, C, H//P, P, W//P, P)
        # N, C, H/P, P, W/P, P -> N*H/P*W/P, P,P, C
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(-1, P, P, C)
        # (BS * SEQ, P,P, C) -> (BS * SEQ, P/SP, SP, P/SP, SP, C) -> (.., P/SP, P/SP, SP*SP*C)
        patches = patches.view(-1, P//SP, SP, P//SP, SP, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, (P//SP)**2, (SP**2)*C)
        subpatch_embeddings = self.subpatch_embedder(patches)  # (N, H*W, D')
        x = self.x_embedder(x)   # (N, T, D), where T = H * W / patch_size ** 2

        N, x_T, D = x.shape
        if cls_token is not None:
            cls_token = self.cls_projectors2(cls_token)
            cls_token = self.wg_norm(cls_token)
            cls_token = cls_token.unsqueeze(1)  # [b, length, d]
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed
        else:
            exit()

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y
        
        x = torch.cat((x, t_embed.unsqueeze(1), y.unsqueeze(1), self.registers.unsqueeze(0).repeat(N, 1, 1)), dim=1)  # (N, T+2+R, D)

        # Split blocks into two halves for skip connections
        half_depth = len(self.blocks) // 2
        skip_activations = []

        aux_losses = 0
        
        # First half: save activations for skip connections (excluding last 4 blocks)
        for i, block in enumerate(self.blocks[:half_depth]):
            x = block(x, c)

            registers = x[:, -self.registers.shape[0]:]  # (N, R, D)
            concepts, info = self.concept_banks[i](registers)
            
            # Compute auxiliary losses
            aux_loss, metrics = compute_concept_losses(
                info, 
                n_concepts=self.n_concepts
            )

            aux_losses += aux_loss

            # Replace registers with concepts
            x = torch.cat((x[:, :-self.registers.shape[0]], concepts), dim=1)

            if self.uvit_skips:
                skip_activations.append(x)
            if (i + 1) == self.encoder_depth:
                projector_in = x[:, :x_T+1]
                zs = [projector(projector_in.reshape(-1, D)).reshape(N, x_T+1, -1) for projector in self.projectors]

        
        # Second half: use skip connections in reverse order
        for i, block in enumerate(self.blocks[half_depth:]):
            if self.uvit_skips:
                # Get corresponding skip activation (in reverse order)
                skip_idx = half_depth - 1 - i
                if skip_idx >= 0 and skip_idx < len(skip_activations):
                    skip_activation = skip_activations[skip_idx]
                    # Concatenate skip connection with current activation
                    x_with_skip = torch.cat([x, skip_activation], dim=-1)
                    # Project back to hidden_size
                    x = self.skip_linears[i](x_with_skip)
                
            x = block(x, c)
            if i != half_depth -1:
                registers = x[:, -self.registers.shape[0]:]  # (N, R, D)
                concepts, info = self.concept_banks[i+half_depth](registers)
                
                # Compute auxiliary losses
                aux_loss, metrics = compute_concept_losses(
                    info, 
                    n_concepts=self.n_concepts
                )

                aux_losses += aux_loss
                
            # Handle pixel calculations for the last 4 blocks
            if i >= len(self.blocks[half_depth:]) - 4:
                pixel_calc_idx = i - (len(self.blocks[half_depth:]) - 4)
                latent = x[:, 1:x_T+1]  # (N, T, D)
                # (N, T, D) -> (N, T, P*P, D)  Repeat the latent p*p times to match pixel embeddings
                latent = latent.unsqueeze(2).repeat(1, 1, (P//SP)**2, 1).reshape(-1, (P//SP)**2, D)
                subpatch_embeddings = self.pixel_calculators[pixel_calc_idx](torch.cat([latent, subpatch_embeddings], dim=-1))
        
        _, cls_token = self.final_layer(x, c, cls=cls_token)
        x = self.nerf_final_layer(subpatch_embeddings)  # (N*H/P*W/P, P//SP*P//4, SP*SP*C)
        # (N*H/P*W/P, P//SP*P//SP, SP*SP*C) -> (N*H/P*W/P, P*P, C)
        x = x.view(-1, (P//SP), (P//SP), SP, SP, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, P**2, C)
        x = x.transpose(1, 2).reshape(N, x_T, -1)
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), (H, W), kernel_size=P, stride=P)

        return x, zs, cls_token, aux_losses


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
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


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_XL_16(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=16, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_L_16(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=16, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_B_16(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def SiT_S_16(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8, 'SiT-XL/16': SiT_XL_16,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8, 'SiT-L/16': SiT_L_16,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8, 'SiT-B/16': SiT_B_16,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8, 'SiT-S/16': SiT_S_16,
}

