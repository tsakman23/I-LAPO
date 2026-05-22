import wandb
import torch
from .nn import IResNetDecoder

def fd_regulariser(decoder: IResNetDecoder,
                   z: torch.Tensor,
                   eps: float = 1e-3) -> torch.Tensor:
    """
    Forward FD regulariser adapted from train_inn_classifier.py:
    encourages small local Lipschitz norms ||J_f(z) v||.
    """
    z32 = z.detach().float()  # decoder only, no IDM gradients

    v = torch.randn_like(z32)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    a = decoder(z32)
    a_pert = decoder(z32 + eps * v)

    local_lip = (a_pert - a).norm(dim=-1) / eps  # approx ||J_f(z) v||
    return local_lip.pow(2).mean()


def decoder_jacobian(decoder: IResNetDecoder,
                     z_single: torch.Tensor) -> torch.Tensor:
    """
    Exact Jacobian J_f(z) for 4-6D decoder, like computeSVDjacobian().
    """
    z_single = z_single.detach().float().requires_grad_(True)
    a = decoder(z_single.unsqueeze(0))  # 1 x d
    out = a[0]                          # d

    d = out.shape[0]
    J = torch.zeros(d, d, device=z_single.device, dtype=z_single.dtype)
    for i in range(d):
        if z_single.grad is not None:
            z_single.grad.zero_()
        grad_i = torch.autograd.grad(
            out[i], z_single,
            retain_graph=True,
            create_graph=False,
        )[0]
        J[i] = grad_i
    return J


def log_decoder_condition(decoder: IResNetDecoder,
                          z_batch: torch.Tensor):
    z0 = z_batch[0]
    J = decoder_jacobian(decoder, z0)
    S = torch.linalg.svdvals(J)
    cond = (S.max() / (S.min() + 1e-8)).item()
    wandb.log({"lapo/decoder_condition": cond})