import math

import torch
import torch.nn as nn
from src.spectral_norm_fc import spectral_norm_fc
from vector_quantize_pytorch import FSQ

from .utils import weight_init
import os

def maybe_chunked_encoder(encoder, x):
    chunk = int(os.getenv("LAOM_ENCODER_CHUNK", "0"))

    if chunk <= 0 or x.shape[0] <= chunk:
        return encoder(x)

    return torch.cat(
        [encoder(x_chunk.contiguous()) for x_chunk in x.split(chunk, dim=0)],
        dim=0,
    )

class MLPBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand * dim),
            nn.ReLU6(),
            nn.Linear(expand * dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class LatentActHead(nn.Module):
    def __init__(self, act_dim, emb_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        self.proj0 = nn.Linear(2 * emb_dim, hidden_dim)
        self.proj1 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, act_dim)

        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, obs_emb, next_obs_emb):
        x = self.block0(self.proj0(torch.concat([obs_emb, next_obs_emb], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.proj_end(x)
        return x


class LatentObsHead(nn.Module):
    def __init__(self, act_dim, proj_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        self.proj0 = nn.Linear(act_dim + proj_dim, hidden_dim)
        self.proj1 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, proj_dim)

        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, x, action):
        x = self.block0(self.proj0(torch.concat([x, action], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, action], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, action], dim=-1)))
        x = self.proj_end(x)
        return x


# inspired by:
# 1. https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py
# 2. https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2, dropout=0.0, downscale=True):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._downscale = downscale
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self._downscale else 1,
        )
        # conv downsampling is faster that maxpool, with same perf
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.conv(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        if self._downscale:
            return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self._out_channels, h, w)


class DecoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # upsample + conv works fine, just slower than conv-transpose
        # also: upsample does not work well with orthogonal init (why?)!
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.conv = nn.ConvTranspose2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=2, stride=2
        )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels) for _ in range(num_res_blocks)])

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, h * 2, w * 2)
    

class InvertibleResBlock(nn.Module):
    """
    One block of i-ResNet: y = x + g(x), bijective iff Lip(g) < 1.
    Enforced via spectral normalization on every linear layer.
    """
    def __init__(self, dim: int, hidden_dim: int = 128, coeff: float = 0.8, n_power_iterations: int = 10):
        super().__init__()
        self.coeff = coeff
        self.g = nn.Sequential(
            spectral_norm_fc(
                nn.Linear(dim, hidden_dim), 
                coeff=coeff, 
                n_power_iterations=n_power_iterations
            ),
            nn.ELU(), # ELU is smooth, continuous gradient, and backprop through fixed-point inverse iter will produce cleaner grads than ReLU.
            spectral_norm_fc(
                nn.Linear(hidden_dim, hidden_dim), 
                coeff=coeff, 
                n_power_iterations=n_power_iterations
            ),
            nn.ELU(),
            spectral_norm_fc(
                nn.Linear(hidden_dim, dim), 
                coeff=coeff, 
                n_power_iterations=n_power_iterations
            ),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = x + g(x), Lip(g) <= coeff < 1 => invertible
        return x + self.g(x)
    
    @torch.no_grad()
    def inverse(self, y: torch.Tensor, max_iter: int = 50, tol: float = 1e-4) -> torch.Tensor:
        """Fixed-point iteration to compute inverse.
        Solves for x in y = x + g(x) => x = y - g(x).
        x_{k+1} = y - g(x_k)
        Contraction with rate Lip(g) <= coeff^(num_layers) < 1, so converges geometrically.
        No gradients: treat as numerical solver only to avoid exploding/vanishing gradients (Behrmann 2019/2021).
        """
        x = y.clone()  # Initial guess
        for _ in range(max_iter):
            x_next = y - self.g(x)
            if (x_next - x).norm(dim=-1).max() < tol:
                return x_next
            x = x_next
        return x


class Actor(nn.Module):
    def __init__(
        self,
        shape,
        num_actions,
        encoder_scale=1,
        encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=1,
        dropout=0.0,
    ):
        super().__init__()
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.final_encoder_shape = shape
        self.encoder = nn.Sequential(
            *conv_stack,
            # nn.Flatten(),
        )
        self.actor_mean = nn.Sequential(
            nn.ReLU6(),
            # works either way...
            # nn.Linear(math.prod(shape), num_actions),
            nn.Linear(shape[0], num_actions),
        )
        self.num_actions = num_actions
        self.apply(weight_init)

    def forward(self, obs):
        out = self.encoder(obs)
        out = out.flatten(2).mean(-1)
        act = self.actor_mean(out)
        return act, out


class ActionDecoder(nn.Module):
    def __init__(self, obs_emb_dim, latent_act_dim, true_act_dim, hidden_dim=128):
        super().__init__()
        self.obs_emb_dim = obs_emb_dim
        self.latent_act_dim = latent_act_dim
        self.true_act_dim = true_act_dim

        self.model = nn.Sequential(
            # nn.Linear(latent_act_dim + obs_emb_dim, hidden_dim)
            nn.Linear(latent_act_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, true_act_dim),
        )

    def forward(self, obs_emb, latent_act):
        # hidden = torch.concat([obs_emb, latent_act], dim=-1)
        # true_act_pred = self.model(hidden)
        true_act_pred = self.model(latent_act)
        return true_act_pred


class IResNetDecoder(nn.Module):
    """
    Bijective decoder f: R^d -> R^d where d = d_z = d_a NECESSARY for invertibility.
    Can be used to decode latent actions into true actions, while still allowing for exact inference of latent actions from true actions via the inverse.
    Composed of n_blocks invertible residual blocks, which are guaranteed to be bijective if the Lipschitz constant of the residual function is less than 1 (enforced via spectral normalization and rescaling).
    
    Args:
        act_dim: Dimension of both latent action space AND true action space.
                 Must equal d_z == d_a.
        hidden_dim: Width of the inner layers of the residual function g in each i-ResNet block (MLP).
        n_blocks: Number of stacked i-ResNet blocks.
        n_power_iterations: Number of power iterations for spectral norm approximation.
    """
    def __init__(self, act_dim: int, 
                 hidden_dim: int = 128, 
                 n_blocks: int = 3, 
                 coeff: float = 0.8, 
                 n_power_iterations: int = 10):
        super().__init__()
        if act_dim <= 0:
            raise ValueError(f"act_dim must be positive, got {act_dim}")
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        self.blocks = nn.ModuleList([
            InvertibleResBlock(
                dim=act_dim, 
                hidden_dim=hidden_dim, 
                coeff=coeff, 
                n_power_iterations=n_power_iterations
            )
            for _ in range(n_blocks)
        ])
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """f(z) -> a"""
        if z.shape[-1] != self.act_dim:
            raise ValueError(
                f"IResNetDecoder is bijective only when d_z == d_a == act_dim; "
                f"got input dim {z.shape[-1]}, expected {self.act_dim}."
            )
        a = z
        for block in self.blocks:
            a = block(a)
        return a
    
    @torch.no_grad()
    def inverse(self, a: torch.Tensor, 
                max_iter: int = 50, 
                tol: float = 1e-4) -> torch.Tensor:
        """f^{-1}(a) -> z via sequential fixed-point inversion."""
        if a.shape[-1] != self.act_dim:
            raise ValueError(
                f"IResNetDecoder is bijective only when d_z == d_a == act_dim; "
                f"got input dim {a.shape[-1]}, expected {self.act_dim}."
            )
        z = a.clone()
        for block in reversed(self.blocks):
            z = block.inverse(z, max_iter=max_iter, tol=tol)
        return z

    def _forward_jacobian_const(self, z_star: torch.Tensor) -> torch.Tensor:
        """Per-sample forward Jacobian J_f(z_star), returned as a CONSTANT (detached).

        Computed in eval() so the spectral-norm hook does not run power iteration
        (no buffer side effects); the normalised weights are identical to train mode.
        Shape: (B, d, d) with J[b, i, j] = d f_i / d z_j at z_star[b].
        """
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=z_star.device.type, enabled=False), torch.enable_grad():
                zin = z_star.detach().float().requires_grad_(True)
                out = self.forward(zin)                      # (B, d)
                d = zin.shape[-1]
                rows = []
                for i in range(d):
                    gi = torch.autograd.grad(
                        out[:, i].sum(), zin, retain_graph=True, create_graph=False
                    )[0]                                     # (B, d)
                    rows.append(gi)
                J = torch.stack(rows, dim=1)                 # (B, d, d)
        finally:
            self.train(was_training)
        return J.detach()

    def inverse_implicit(self, a: torch.Tensor,
                         max_iter: int = 100,
                         tol: float = 1e-4) -> torch.Tensor:
        """Differentiable inverse via the implicit function theorem.

        The fixed-point solve for z* = f^{-1}(a) is done WITHOUT grad, then a single
        Newton-style correction term carries the exact first-order gradient:

            z_out = z* - J_f(z*)^{-1} ( f(z*) - a )

        With z* and J_f(z*) treated as constants and f(z*) / a differentiable, this
        yields  dz/da = J_f^{-1}  and  dz/dtheta = -J_f^{-1} (d f/d theta)  exactly --
        i.e. the implicit gradient -- with NO backprop through the iteration. The
        (well-conditioned) linear solve replaces the unrolled product of Jacobians,
        avoiding the exploding/vanishing gradients of differentiating the fixed point.

        NOTE: for the pure cycle f^{-1}(f(z)) this gradient is ~0 by construction
        (the exact-inverse identity has zero derivative); the signal lives in losses
        where `a` is an independent target, e.g. f^{-1}(a_true) vs predicted latent.
        """
        # Run entirely in fp32 with autocast disabled: the fixed-point solve, the
        # Jacobian and the linear solve need full precision, and bf16 leaking in here
        # corrupts the backward pass ("Found dtype BFloat16 but expected Float").
        with torch.autocast(device_type=a.device.type, enabled=False):
            a = a.float()
            with torch.no_grad():
                z_star = self.inverse(a, max_iter=max_iter, tol=tol).detach()

            J = self._forward_jacobian_const(z_star)         # (B, d, d), constant

            # eval() => differentiable wrt params (grad flows through W_orig) with no
            # power-iteration side effects; z_star is constant, a carries grad.
            was_training = self.training
            self.eval()
            try:
                residual = self.forward(z_star) - a          # (B, d), differentiable
            finally:
                self.train(was_training)

            correction = torch.linalg.solve(J, residual.unsqueeze(-1)).squeeze(-1)
            return z_star - correction


# IDM: (s_t, s_t+1) -> a_t
class IDM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim),
            # nn.LayerNorm(latent_act_dim),
        )

    def forward(self, obs, next_obs):
        # [B, C, H, W] -> [B, 2 * C, H, W]
        concat_obs = torch.concat([obs, next_obs], axis=1)
        latent_action = self.encoder(concat_obs)
        return latent_action


# FDM: (s_t, a_t) -> s_t+1
class FDM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(*conv_stack)
        self.final_encoder_shape = shape

        # decoder
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels[::-1]:
            conv_seq = DecoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.decoder = nn.Sequential(
            *conv_stack,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * encoder_scale, self.inital_shape[0], kernel_size=1),
            nn.Tanh(),
        )
        self.act_proj = nn.Linear(latent_act_dim, math.prod(self.final_encoder_shape))

    def forward(self, obs, latent_action):
        assert obs.ndim == 4, "expect shape [B, C, H, W]"
        obs_emb = self.encoder(obs)
        act_emb = self.act_proj(latent_action).reshape(-1, *self.final_encoder_shape)
        # concat across channels, [B, C * 2, 1, 1]
        emb = torch.concat([obs_emb, act_emb], dim=1)
        next_obs = self.decoder(emb)
        return next_obs


class LAPO(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        self.idm = IDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action

    @torch.no_grad()
    def label(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        return latent_action


# Not used in final experiments, here just for reference.
class IDMFSQ(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim=128,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        fsq_levels=(2, 2),
    ):
        super().__init__()
        assert latent_act_dim % len(fsq_levels) == 0
        self.latent_act_dim = latent_act_dim
        self.fsq_levels = fsq_levels

        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim),
            # nn.LayerNorm(latent_act_dim),
        )
        self.quantizer = FSQ(levels=list(fsq_levels))

    def forward(self, obs, next_obs):
        # [B, C, H, W] -> [B, 2 * C, H, W]
        concat_obs = torch.concat([obs, next_obs], axis=1)
        # [B, la_dim]
        latent_action = self.encoder(concat_obs)
        # [B, la_split, la_dim // la_split]
        latent_action = latent_action.reshape(latent_action.shape[0], self.latent_act_dim // len(self.fsq_levels), -1)
        quantized_latent_action, indices = self.quantizer(latent_action)
        quantized_latent_action = quantized_latent_action.reshape(concat_obs.shape[0], -1)
        assert quantized_latent_action.shape[-1] == self.latent_act_dim

        return quantized_latent_action


class LAPOFSQ(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim=128,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        fsq_levels=(2, 2, 4),
    ):
        super().__init__()
        self.idm = IDMFSQ(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
            fsq_levels=fsq_levels,
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action

    @torch.no_grad()
    def label(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        return latent_action


class LAOM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.act_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.obs_head = LatentObsHead(latent_act_dim, math.prod(shape), obs_head_dim, dropout=obs_head_dropout)
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        latent_next_obs = self.obs_head(obs_emb.flatten(1).detach(), latent_action)

        return latent_next_obs, latent_action, obs_emb.detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action


class LAOMWithLabels(nn.Module):
    def __init__(
        self,
        shape,
        true_act_dim,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.idm_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.true_actions_head = nn.Linear(latent_act_dim, true_act_dim)

        self.fdm_head = LatentObsHead(latent_act_dim, math.prod(shape), obs_head_dim, dropout=obs_head_dropout)
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs, predict_true_act=False):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        # obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        obs_next = torch.concat([obs, next_obs], dim=0)
        obs_emb, next_obs_emb = maybe_chunked_encoder(self.encoder, obs_next).split(obs.shape[0])
        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        latent_next_obs = self.fdm_head(obs_emb.flatten(1).detach(), latent_action)
        # TODO: use norm from encoder here too!

        if predict_true_act:
            true_action = self.true_actions_head(latent_action)
            return latent_next_obs, latent_action, true_action, obs_emb.flatten(1).detach()

        return latent_next_obs, latent_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        # obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        obs_next = torch.concat([obs, next_obs], dim=0)
        obs_emb, next_obs_emb = maybe_chunked_encoder(self.encoder, obs_next).split(obs.shape[0])
        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action
    
    
class LAOMWithLabelsInvertible(LAOMWithLabels):
    def __init__(
        self,
        inv_stage,
        shape,
        true_act_dim,
        latent_act_dim,
        ires_hidden_dim=128,
        ires_n_blocks=3,
        ires_n_power_iter=10,
        ires_coeff=0.8,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__(
            shape=shape,
            true_act_dim=true_act_dim,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
            encoder_dropout=encoder_dropout,
            encoder_norm_out=encoder_norm_out,
            act_head_dim=act_head_dim,
            act_head_dropout=act_head_dropout,
            obs_head_dim=obs_head_dim,
            obs_head_dropout=obs_head_dropout,
        )
        
        # Checks
        if inv_stage not in [0, 1, 3]:
            raise ValueError(f"inv_stage must be one of [0, 1, 3], got {inv_stage}. Select 0 for full invertibility, 1 for Stage 1, and 3 for Stage 3 (0 default).")
        
        if inv_stage in [0, 1]:
            if latent_act_dim != true_act_dim:
                raise ValueError(f"For invertibility, latent_act_dim must equal true_act_dim, got {latent_act_dim} and {true_act_dim}")

            self.true_actions_head = IResNetDecoder(
                act_dim=true_act_dim,
                hidden_dim=ires_hidden_dim,
                n_blocks=ires_n_blocks,
                coeff=ires_coeff,
                n_power_iterations=ires_n_power_iter,
            )
        else:
            self.true_actions_head = nn.Linear(latent_act_dim, true_act_dim)
            
        # Note: weight_init from parent __init__ was applied to discarded nn.Linear.
        # IResNetDecoder is intentionally left with default PyTorch init +
        # spectral normalization - do not apply weight_init here.


class IDMLabels(nn.Module):
    def __init__(
        self,
        shape,
        act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        act_head_dim=512,
        act_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            # nn.LayerNorm(math.prod(shape))
        )
        self.idm_head = LatentActHead(act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)

        self.act_dim = act_dim
        self.final_encoder_shape = shape
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        pred_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))

        return pred_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        pred_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return pred_action
