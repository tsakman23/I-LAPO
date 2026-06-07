# LAOM+supervision

## 1. Data Collection (DCS)

[x] Hopper-hop unlabeled:

```bash
python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/hopper-hop-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --save_path="/data2/laom/data/dcs/hopper/hopper-hop-dcs-train-5000traj.hdf5" \
    --num_trajectories=5000 \
    --dcs_difficulty="scale_easy_video_hard" \
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=True
```

[x] Hopper-hop labeled:

```bash
python -m scripts.sample_labeled_data --data_path="/data2/laom/data/dcs/hopper/hopper-hop-dcs-train-5000traj.hdf5" --save_path="/data2/laom/data/dcs/hopper/hopper-hop-dcs-labeled-125traj.hdf5" --chunk_size=1000 --num_trajectories=125
```

[x] Hopper-hop eval:

```bash
python -m scripts.create_eval_labeled_data \
  --data_path="/data2/laom/data/dcs/hopper/hopper-hop-dcs-train-5000traj.hdf5" \
  --save_path="/data2/laom/data/dcs/hopper/hopper-hop-dcs-eval-labeled-50traj.hdf5" \
  --train_path="/data2/laom/data/dcs/hopper/hopper-hop-dcs-labeled-125traj.hdf5" \
  --chunk_size=1000 \
  --num_trajectories=50 \
  --seed_start=0
```

[x] Cheetah-run unlabeled:

```bash
python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/cheetah-run-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --save_path="/data2/laom/data/dcs/cheetah/cheetah-run-dcs-train-5000traj.hdf5" \
    --num_trajectories=5000 \
    --dcs_difficulty="scale_easy_video_hard" \
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=True
```

[x] Cheetah-run labeled:

```bash
python -m scripts.sample_labeled_data --data_path="/data2/laom/data/dcs/cheetah/cheetah-run-dcs-train-5000traj.hdf5" --save_path="/data2/laom/data/dcs/cheetah/cheetah-run-dcs-labeled-125traj.hdf5" --chunk_size=1000 --num_trajectories=125
```

[x] Cheetah-run eval:

```bash
python -m scripts.create_eval_labeled_data \
  --data_path="/data2/laom/data/dcs/cheetah/cheetah-run-dcs-train-5000traj.hdf5" \
  --save_path="/data2/laom/data/dcs/cheetah/cheetah-run-dcs-eval-labeled-50traj.hdf5" \
  --train_path="/data2/laom/data/dcs/cheetah/cheetah-run-dcs-labeled-125traj.hdf5" \
  --chunk_size=1000 \
  --num_trajectories=50 \
  --seed_start=0
```

[x] Walker-run unlabeled:

```bash
python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/walker-run-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --save_path="/data2/laom/data/dcs/walker/walker-run-dcs-train-5000traj.hdf5" \
    --num_trajectories=5000 \
    --dcs_difficulty="scale_easy_video_hard" \
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=True
```

[x] Walker-run labeled:

```bash
python -m scripts.sample_labeled_data --data_path="/data2/laom/data/dcs/walker/walker-run-dcs-train-5000traj.hdf5" --save_path="/data2/laom/data/dcs/walker/walker-run-dcs-labeled-125traj.hdf5" --chunk_size=1000 --num_trajectories=125
```

[x] Walker-run eval:

```bash
python -m scripts.create_eval_labeled_data \
  --data_path="/data2/laom/data/dcs/walker/walker-run-dcs-train-5000traj.hdf5" \
  --save_path="/data2/laom/data/dcs/walker/walker-run-dcs-eval-labeled-50traj.hdf5" \
  --train_path="/data2/laom/data/dcs/walker/walker-run-dcs-labeled-125traj.hdf5" \
  --chunk_size=1000 \
  --num_trajectories=50 \
  --seed_start=0
```

## 2. LAOM+supervision reproduction for each task
- Repeat 3 times with different seeds

[x] Cheetah:

```bash
python -m train_laom_labels --config_path=configs/step0-reproduce-laom+supervision/step0-cheetah-dcs-replicate.yaml
```

[x] Hopper:

```bash
python -m train_laom_labels --config_path=configs/step0-reproduce-laom+supervision/step0-hopper-dcs-replicate.yaml
```

[x] Walker:

```bash
python -m train_laom_labels --config_path=configs/step0-reproduce-laom+supervision/step0-walker-dcs-replicate.yaml
```

## 3. Ablation 1. Decoder architecture (3 seeds)
- Cheetah only
- labeled_loss_coef=0.001
- Metrics:
  - Decoder Eval episodic normalized return (mean +- std from different seed runs)
  - Decoder Reconstruction error
  - Cycle consistency error (i-ResNets only, and ONLY as a diagnostic, no gradient backpropagation)
  - Linear action decoder probe R^2
  - Decoder Jacobian condition number for cases 3 and 5

| Condition | Stage 1 | Stage 3 | Latent_action_dim |
| --- | --- | --- | --- |
| 1. LAOM+supervision | Linear | MLP | 8192 |
| 2. LAOM+supervision | Linear | MLP | 6 |
| 3. I-LAPO-S1 | i-ResNet | MLP (fresh) | 6 |
| 4. I-LAPO-S3 | Linear | i-ResNet (fresh) | 6 |
| 5. I-LAPO-full | i-ResNet | i-ResNet (REUSED) | 6 |

- Obtain best variant -> I-LAPO*, use in all subsequent experiments and ablations.
- Run LAOM+supervision at both d_z = 8192 (its native setting, reported as an upper-bound reference) and d_z = 6 (the controlled comparison). Then run all I-LAPO variants at d_z = 6. This gives me:
  - A fair architectural comparison at d_z = 6 across all four conditions
  - An honest upper-bound reference showing what LAOM+supervision achieves at its intended dimensionality

[x] Case 1/5:

Reuse results from reproduction runs

[] Case 2/5:

Same script (`train_laom_labels.py`) as cheetah reproduction (but this time latent_action_dim=6):

```bash
python -m train_laom_labels --config_path=configs/abl1-decoder-arch/laom-6-cheetah-dcs.yaml --seed=0
```

[] Case 3/5:

From here onwards we use `train_laom_labels_inv.py`, not `train_laom_labels.py`, as it is equipped to handle the i-ResNet logic.

```bash
python -m train_laom_labels_inv --config_path=configs/abl1-decoder-arch/s1-inv-cheetah-dcs.yaml --seed=0
```

[] Case 4/5:

```bash
python -m train_laom_labels_inv --config_path=configs/abl1-decoder-arch/s3-inv-cheetah-dcs.yaml --seed=0
```

[] Case 5/5:

```bash
python -m train_laom_labels_inv --config_path=configs/abl1-decoder-arch/full-inv-cheetah-dcs.yaml --seed=0
```

