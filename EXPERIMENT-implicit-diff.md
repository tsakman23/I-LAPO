# Experiment: implicit-function-theorem differentiable inverse

Branch `experiment/implicit-diff-cycle`, isolated git worktree at `/data2/laom-implicit-diff`.
Built on a snapshot of `main`'s WIP (first commit) + the experiment changes (this commit).
`main` is untouched; the WIP is also backed up in `git stash` (`stash@{0}`).

## What changed
- `src/nn.py` — `IResNetDecoder.inverse_implicit(a)` + helper `_forward_jacobian_const`.
  Differentiable inverse via the IFT: solve the fixed point with no grad, then carry
  the exact first-order gradient through one correction term
  `z = z* - J_f(z*)^{-1} (f(z*) - a)`. No backprop through the iteration.
- `train_laom_labels_inv.py` — opt-in `cycle_grad_mode: "none"|"implicit"` config + a
  differentiable cycle term (used only when `cycle_grad_mode="implicit"` and
  `cycle_loss_coef>0`) + a `lapo/cycle_loss_term` wandb metric. Default = unchanged.
- `check_cycle_grad.py` — standalone CPU micro-benchmark: unrolled vs implicit gradients.

## Run the test (seconds, no training)
```bash
cd /data2/laom-implicit-diff
/data2/laom/conda_env/bin/python check_cycle_grad.py
# stress the Lipschitz cap: --weight_scale 50 ; constraint off: --coeffs 0.9,1.0,1.6 --weight_scale 50
```

## Enable in a training run
Set in the config / CLI: `--lapo.cycle_grad_mode implicit --lapo.cycle_loss_coef <e.g. 0.1>`.

## Findings (from check_cycle_grad.py)
1. **Implicit gradient is exact** — matches the unrolled gradient (ratio 1.00) wherever
   unrolling is valid, across coeff in [0.5, 0.999], at init and saturated weights.
2. **Implicit is ~20x cheaper** (no deep grad graph), so no training-time concern.
3. **The pure cycle loss `||f^{-1}(f(z)) - z||^2` is vacuous**: its exact gradient is ~0
   (the inverse identity has zero derivative). The explosion seen when *unrolling* it was
   numerical noise amplified through a deep graph, not real signal.
4. **The i-ResNet inverse is not the explosion source.** Lip(g)<1 (realized, via ELU
   slopes + spectral cap) keeps the fixed point contractive even at coeff>=1, so unrolling
   converges to the implicit value. Pipeline explosions come from elsewhere (bf16, the
   encoder coupling, or the vacuous cycle term) — worth stating explicitly in the report.

## Cleanup when done
```bash
git -C /data2/laom worktree remove /data2/laom-implicit-diff   # add --force if PDF/untracked remain
git -C /data2/laom branch -D experiment/implicit-diff-cycle    # only if you don't want to keep it
git -C /data2/laom stash drop stash@{0}                        # only once you've confirmed main is fine
```
