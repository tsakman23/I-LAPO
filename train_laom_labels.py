import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from sklearn.metrics import r2_score

from src.augmentations import Augmenter
from src.checkpoint import load_model, save_model, save_run_config, stage_path
from src.configs import BCConfig, DecoderConfig, LAOMConfigBase
from src.nn import ActionDecoder, Actor, LAOMWithLabels
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSInMemoryDataset,
    DCSLAOMInMemoryDataset,
    DCSLAOMTrueActionsDataset,
    create_env_from_df,
    get_bc_normalizer,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    soft_update,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LAOMConfig(LAOMConfigBase):
    pass


@dataclass
class Config:
    project: str = "laom"
    group: str = "laom-labels"
    name: str = "laom-labels"
    seed: int = 0

    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"
    # path to a saved Stage-1 LAM checkpoint; if set, skip Stage 1 and run only
    # Stage 2 (BC) + Stage 3 (decoder). Lets ablations reuse the ~14h LAM.
    resume_lam_from: Optional[str] = None

    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-la_dim={self.lapo.latent_action_dim}-lab_loss_coef={self.lapo.labeled_loss_coef}-seed={self.seed}-{uuid.uuid4().hex[:6]}"
        # coupling labeled dataset for laom pretraining and action decoder finetuning
        self.decoder.data_path = self.lapo.labeled_data_path


@torch.no_grad()
def evaluate(lam, dataloader, device):
    lam.eval()
    total_samples, total_loss = 0, 0.0

    for batch in dataloader:
        obs, next_obs, _, actions, _, _ = [b.to(device) for b in batch]
        obs = normalize_img(obs.permute((0, 3, 1, 2)))
        next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

        with torch.autocast(device, dtype=torch.bfloat16):
            _, _, pred_action, _ = lam(obs, next_obs, predict_true_act=True)
            eval_loss = F.mse_loss(pred_action, actions, reduction="sum")

        total_loss += eval_loss.item()
        total_samples += obs.shape[0]

    lam.train()
    return total_loss / total_samples


def train_laom(config: LAOMConfig):
    dataset = DCSLAOMInMemoryDataset(
        config.data_path, max_offset=config.future_obs_offset, frame_stack=config.frame_stack, device=DEVICE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    labeled_dataset = DCSLAOMTrueActionsDataset(
        config.labeled_data_path,
        max_offset=config.future_obs_offset,
        frame_stack=config.frame_stack,
        device=DEVICE,
    )
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=config.labeled_batch_size)

    if config.eval_data_path is not None:
        eval_dataset = DCSLAOMInMemoryDataset(
            config.eval_data_path, max_offset=1, frame_stack=config.frame_stack, device=DEVICE
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

    lapo_kwargs = dict(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        true_act_dim=dataset.act_dim,
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
    )
    lapo = LAOMWithLabels(**lapo_kwargs).to(DEVICE)
    lapo._build_kwargs = lapo_kwargs

    target_lapo = deepcopy(lapo)
    for p in target_lapo.parameters():
        p.requires_grad_(False)

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    optim = torch.optim.Adam(
        params=get_optim_groups(lapo, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    augmenter = Augmenter(dataset.img_hw)

    state_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)

    print("Final encoder shape:", math.prod(lapo.final_encoder_shape))
    state_act_linear_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_iterations = 0
    total_tokens = 0

    labeled_dataloader_iter = iter(labeled_dataloader)
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        for i, batch in enumerate(tqdm(dataloader, desc="Batches", leave=False)):
            total_tokens += config.batch_size
            total_iterations += 1

            obs, next_obs, future_obs, debug_actions, debug_states, _ = [b.to(DEVICE) for b in batch]

            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))
            future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                obs_aug = augmenter(obs)
                future_obs_aug = augmenter(future_obs)
                next_obs_aug = augmenter(next_obs)

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                if config.use_aug:
                    # using augmenter directly will not work due to bf16
                    latent_next_obs, latent_action, obs_hidden = lapo(obs_aug, future_obs_aug)
                else:
                    latent_next_obs, latent_action, obs_hidden = lapo(obs, future_obs)

                with torch.no_grad():
                    if config.use_aug:
                        next_obs_target = target_lapo.encoder(next_obs_aug).flatten(1)
                    else:
                        next_obs_target = target_lapo.encoder(next_obs).flatten(1)

                if config.cosine_loss:
                    loss0 = 1 - F.cosine_similarity(latent_next_obs, next_obs_target.detach(), dim=-1).mean()
                else:
                    loss0 = F.mse_loss(latent_next_obs, next_obs_target.detach())

            # loss with true actions
            labeled_batch = next(labeled_dataloader_iter)
            label_obs, label_next_obs, label_future_obs, label_actions, _, _ = [b.to(DEVICE) for b in labeled_batch]

            label_obs = normalize_img(label_obs.permute((0, 3, 1, 2)))
            label_future_obs = normalize_img(label_future_obs.permute((0, 3, 1, 2)))
            label_next_obs = normalize_img(label_next_obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                label_obs_aug = augmenter(label_obs)
                label_future_obs_aug = augmenter(label_future_obs)

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                if config.use_aug:
                    # using augmenter directly will not work due to bf16
                    _, _, pred_action, _ = lapo(label_obs_aug, label_future_obs_aug, predict_true_act=True)
                else:
                    _, _, pred_action, _ = lapo(label_obs, label_future_obs, predict_true_act=True)

                loss1 = F.mse_loss(pred_action, label_actions)

            loss = loss0 + config.labeled_loss_coef * loss1

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            if i % config.target_update_every == 0:
                soft_update(target_lapo, lapo, tau=config.target_tau)

            # update state probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, debug_states)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, debug_actions)

            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()
            
            act_probe_r2 = None
            if total_iterations % 100 == 0:
                # R^2 - computed outside autocast to stay in float32 for sklearn
                _pred = pred_action.detach().float().cpu().numpy()
                _true = debug_actions.detach().float().cpu().numpy()
                act_probe_r2 = r2_score(_true, _pred, multioutput="uniform_average")

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, debug_actions)

            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            log_data = {
                "lapo/total_loss": loss.item(),
                "lapo/mse_loss": loss0.item(),
                "lapo/true_action_mse_loss": loss1.item(),
                "lapo/state_probe_mse_loss": state_probe_loss.item(),
                "lapo/action_probe_mse_loss": act_probe_loss.item(),
                "lapo/state_action_probe_mse_loss": state_act_probe_loss.item(),
                "lapo/throughput": total_tokens / (time.time() - start_time),
                "lapo/learning_rate": scheduler.get_last_lr()[0],
                "lapo/grad_norm": get_grad_norm(lapo).item(),
                "lapo/target_obs_norm": torch.norm(next_obs_target, p=2, dim=-1).mean().item(),
                "lapo/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                "lapo/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                "lapo/epoch": epoch,
                "lapo/total_steps": total_iterations,
            }
            if act_probe_r2 is not None:
                log_data["lapo/action_probe_r2"] = act_probe_r2
            wandb.log(log_data)

        if config.eval_data_path is not None:
            eval_mse_loss = evaluate(lapo, eval_dataloader, device=DEVICE)
            wandb.log(
                {
                    "lapo/eval_true_action_mse_loss": eval_mse_loss,
                    "lapo/epoch": epoch,
                    "lapo/total_steps": total_iterations,
                }
            )

    return lapo


@torch.no_grad()
def evaluate_bc(env, actor, num_episodes, seed=0, device="cpu", action_decoder=None):
    returns = []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None].permute(0, 3, 1, 2)
            obs_ = normalize_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                if isinstance(action_decoder, ActionDecoder):
                    action = action_decoder(obs_emb, action)
                else:
                    action = action_decoder(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def train_bc(lam: LAOMWithLabels, config: BCConfig):
    dataset = DCSInMemoryDataset(config.data_path, frame_stack=config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor_kwargs = dict(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    )
    actor = Actor(**actor_kwargs).to(DEVICE)
    actor._build_kwargs = actor_kwargs

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    # for debug
    print("Latent action dim:", num_actions)
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, dataset.act_dim)
    ).to(DEVICE)

    act_decoder_optim = torch.optim.AdamW(params=act_decoder.parameters(), lr=config.learning_rate, fused=True)
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

            # label with lapo latent actions
            target_actions = lam.label(obs, next_obs)

            # augment obs only for bc to make action labels determenistic
            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            # optimizing the probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                decoder_loss = F.mse_loss(pred_true_actions, true_actions)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps,
                }
            )

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=act_decoder,
    )
    wandb.log(
        {
            "bc/eval_returns_mean": eval_returns.mean(),
            "bc/eval_returns_std": eval_returns.std(),
            "bc/epoch": epoch,
            "bc/total_steps": total_steps,
        }
    )

    return actor


def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig):
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dataset = DCSInMemoryDataset(config.data_path, frame_stack=bc_config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    # to make equal number of updates for all labeled datasets which vary in size
    num_epochs = config.total_updates // len(dataloader)

    action_decoder_kwargs = dict(
        obs_emb_dim=math.prod(actor.final_encoder_shape),
        latent_act_dim=actor.num_actions,
        true_act_dim=dataset.act_dim,
        hidden_dim=config.hidden_dim,
    )
    action_decoder = ActionDecoder(**action_decoder_kwargs).to(DEVICE)
    action_decoder._build_kwargs = action_decoder_kwargs

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, config.weight_decay), lr=config.learning_rate, fused=True
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=bc_config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    # scheduler setup
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0

    for epoch in trange(num_epochs, desc="Epochs"):
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                with torch.no_grad():
                    latent_actions, obs_emb = actor(obs)
                pred_actions = action_decoder(obs_emb, latent_actions)

                loss = F.mse_loss(pred_actions, true_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            wandb.log(
                {
                    "decoder/mse_loss": loss.item(),
                    "decoder/throughput": total_tokens / (time.time() - start_time),
                    "decoder/learning_rate": scheduler.get_last_lr()[0],
                    "decoder/epoch": epoch,
                    "decoder/total_steps": total_steps,
                }
            )

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=action_decoder,
    )
    eval_log = {
        "decoder/eval_returns_mean": eval_returns.mean(),
        "decoder/eval_returns_std": eval_returns.std(),
        "decoder/epoch": epoch,
        "decoder/total_steps": total_steps,
    }
    norm_value = get_bc_normalizer(config.data_path)
    if norm_value is not None:
        norm_returns = eval_returns / norm_value
        eval_log["decoder/eval_returns_norm_mean"] = norm_returns.mean()
    wandb.log(eval_log)

    return action_decoder


@pyrallis.wrap()
def train(config: Config):
    tags = []
    data_path_lower = config.lapo.data_path.lower()
    for env_name in ("cheetah", "hopper", "walker"):
        if env_name in data_path_lower:
            tags.append(env_name)
    tags.extend(
        [
            f"seed={config.seed}",
            f"labeled_loss_coef={config.lapo.labeled_loss_coef}",
            f"la_dim={config.lapo.latent_action_dim}",
        ]
    )
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        tags=tags,
        config=asdict(config),
        save_code=True,
    )
    set_seed(config.seed)
    print(f"Device: {DEVICE}")
    if config.save_checkpoints:
        save_run_config(config.checkpoint_dir, config.name, config)
    # stage 1: pretraining lapo on unlabeled dataset (or reuse a saved LAM)
    if config.resume_lam_from is not None:
        print(f"Resuming: loading Stage-1 LAM from {config.resume_lam_from} (skipping Stage 1)")
        lapo, _ = load_model(config.resume_lam_from, map_location=DEVICE, eval_mode=True)
    else:
        lapo = train_laom(config=config.lapo)
        if config.save_checkpoints:
            save_model(stage_path(config.checkpoint_dir, config.name, "stage1_lam.pt"), lapo)
            print(f"Saved stage 1 LAM to {stage_path(config.checkpoint_dir, config.name, 'stage1_lam.pt')}")
    # stage 2: pretraining bc on latent actions
    actor = train_bc(lam=lapo, config=config.bc)
    if config.save_checkpoints:
        save_model(stage_path(config.checkpoint_dir, config.name, "stage2_actor.pt"), actor)
        print(f"Saved stage 2 actor to {stage_path(config.checkpoint_dir, config.name, 'stage2_actor.pt')}")
    # stage 3: finetune on labeles ground-truth actions
    action_decoder = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc)
    if config.save_checkpoints:
        save_model(stage_path(config.checkpoint_dir, config.name, "stage3_decoder.pt"), action_decoder)
        print(f"Saved stage 3 decoder to {stage_path(config.checkpoint_dir, config.name, 'stage3_decoder.pt')}")

    run.finish()
    return lapo, actor, action_decoder


if __name__ == "__main__":
    train()
