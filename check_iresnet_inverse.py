import argparse

import torch

from src.nn import IResNetDecoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check i-ResNet inverse error and Jacobian condition number."
    )
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--coeff", type=float, default=0.8)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_power_iterations", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--cond_samples", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _jacobian(decoder: IResNetDecoder, z_single: torch.Tensor) -> torch.Tensor:
    z_single = z_single.detach().float().requires_grad_(True)
    out = decoder(z_single.unsqueeze(0))[0]
    dim = out.shape[0]
    jac = torch.zeros(dim, dim, device=z_single.device, dtype=z_single.dtype)
    for i in range(dim):
        if z_single.grad is not None:
            z_single.grad.zero_()
        grad_i = torch.autograd.grad(
            out[i], z_single, retain_graph=True, create_graph=False
        )[0]
        jac[i] = grad_i
    return jac


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    decoder = IResNetDecoder(
        act_dim=args.act_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        coeff=args.coeff,
        n_power_iterations=args.n_power_iterations,
    ).to(device)

    decoder.train()
    for _ in range(args.warmup_steps):
        decoder(torch.randn(8, args.act_dim, device=device))
    decoder.eval()

    with torch.no_grad():
        x = torch.randn(args.samples, args.act_dim, device=device)
        y = decoder(x)
        x_hat = decoder.inverse(y, max_iter=args.max_iter, tol=args.tol)
        inv_error = (x_hat - x).norm(dim=-1)

    inv_mean = inv_error.mean().item()
    inv_max = inv_error.max().item()

    conds = []
    for _ in range(args.cond_samples):
        z = torch.randn(args.act_dim, device=device)
        jac = _jacobian(decoder, z)
        svals = torch.linalg.svdvals(jac)
        cond = (svals.max() / (svals.min() + 1e-8)).item()
        conds.append(cond)

    cond_mean = sum(conds) / len(conds) if conds else float("nan")
    cond_max = max(conds) if conds else float("nan")

    print("Decoder settings:")
    print(f"  act_dim={args.act_dim}")
    print(f"  coeff={args.coeff}")
    print(f"  n_blocks={args.n_blocks}")
    print(f"  n_power_iterations={args.n_power_iterations}")
    print(f"  max_iter={args.max_iter}")
    print(f"  tol={args.tol}")

    print("\nInverse error ||x - f^{-1}(f(x))||:")
    print(f"  mean={inv_mean:.6e}")
    print(f"  max ={inv_max:.6e}")
    print(f"  pass (<1e-4): {inv_max < 1e-4}")

    print("\nJacobian condition number kappa:")
    print(f"  mean={cond_mean:.6f}")
    print(f"  max ={cond_max:.6f}")
    print(f"  pass (<100): {cond_max < 100.0}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
