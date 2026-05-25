import wandb
import torch
from .nn import IResNetDecoder

def fd_regulariser(decoder: IResNetDecoder,
                   z: torch.Tensor,
                   eps: float = 0.1) -> torch.Tensor:
    """
    Forward FD regulariser adapted from train_inn_classifier.py:
    encourages small local Lipschitz norms ||J_f(z) v||.
    """
    z32 = z.detach().float()  # Detach z from the IDM/FDM comp. graph, pass FD gradients only through decoder weights.

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

    d_a = out.shape[0]
    d_z = z_single.shape[0]
    J = torch.zeros(d_a, d_z, device=z_single.device, dtype=z_single.dtype) # d_a = d_z = d
    for i in range(d_a):
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
                          z_batch: torch.Tensor,
                          n_samples: int = 4):
    indices = torch.randperm(len(z_batch))[:n_samples]
    conds = []
    for idx in indices:
        J = decoder_jacobian(decoder, z_batch[idx])
        with torch.no_grad():
            S = torch.linalg.svdvals(J)
        conds.append((S.max() / (S.min() + 1e-8)).item())
    wandb.log({
        "lapo/decoder_condition_mean": sum(conds) / len(conds),
        "lapo/decoder_condition_max":  max(conds),
    })