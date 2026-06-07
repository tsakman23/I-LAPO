# LAOM+supervision

## 1. Data Collection (Vanilla DMControl)

[x] Cheetah-run unlabeled:

```bash
python -m scripts.data_collection.collect_data --checkpoint_path="scripts/data_collection/checkpoints/cheetah-run-expert" --checkpoint_name="checkpoint.pt"     --dcs_backgrounds_path="DAVIS/JPEGImages/480p" --save_path="/data2/laom/data/cheetah-vanilla-5000traj.hdf5" --num_trajectories=5000 --dcs_difficulty="vanilla" --dcs_backgrounds_split="train" --dcs_img_hw=64 --seed=0 --cuda=True
```

[x] Cheetah-run labeled:

```bash
python -m scripts.sample_labeled_data --data_path="/data2/laom/data/cheetah-vanilla-5000traj.hdf5" --save_path="/data2/laom/data/cheetah-vanilla-labeled-125traj.hdf5" --chunk_size=1000 --num_trajectories=125
```

[x] Hopper-hop unlabeled:

```bash
python -m scripts.data_collection.collect_data --checkpoint_path="scripts/data_collection/checkpoints/hopper-hop-expert" --checkpoint_name="checkpoint.pt" --dcs_backgrounds_path="DAVIS/JPEGImages/480p" --save_path="/data2/laom/data/hopper-vanilla-5000traj.hdf5" --num_trajectories=5000 --dcs_difficulty="vanilla" --dcs_backgrounds_split="train" --dcs_img_hw=64 --seed=0 --cuda=True
```

[x] Hopper-hop labeled:

```bash
python -m scripts.sample_labeled_data --data_path="/data2/laom/data/hopper-vanilla-5000traj.hdf5" --save_path="/data2/laom/data/hopper-vanilla-labeled-125traj.hdf5" --chunk_size=1000 --num_trajectories=125
```

[x] Walker-run unlabeled:

```bash
python -m scripts.data_collection.collect_data --checkpoint_path="scripts/data_collection/checkpoints/walker-run-expert" --checkpoint_name="checkpoint.pt" --dcs_backgrounds_path="DAVIS/JPEGImages/480p" --save_path="/data2/laom/data/walker-vanilla-5000traj.hdf5" --num_trajectories=5000 --dcs_difficulty="vanilla" --dcs_backgrounds_split="train" --dcs_img_hw=64 --seed=0 --cuda=True
```

[x] Walker-run labeled:

```bash
python -m scripts.sample_labeled_data --data_path="/data2/laom/data/walker-vanilla-5000traj.hdf5" --save_path="/data2/laom/data/walker-vanilla-labeled-125traj.hdf5" --chunk_size=1000 --num_trajectories=125
```

## 2. LAOM+supervision reproduction for each task

[o] Cheetah:

```bash
python -m train_laom_labels --config_path=configs/step0-reproduce-laom+supervision/step0-cheetah-vanilla-replicate.yaml
```

[o] Hopper:

```bash
python -m train_laom_labels --config_path=configs/step0-reproduce-laom+supervision/step0-hopper-vanilla-replicate.yaml
```

[o] Walker:

```bash
python -m train_laom_labels --config_path=configs/step0-reproduce-laom+supervision/step0-walker-vanilla-replicate.yaml
```

## 3. Ablation 1. Decoder architecture (3 seeds)
- Hopper only
- labeled_loss_coef=0.01
- !!! latent_action_dim=6

### 1. LAOM+supervision - S1 Linear - S3 MLP

[] Case 1/4:

Same script (`train_laom_labels.py`) as hopper reproduction (but this time latent_action_dim=6):

```bash
python -m train_laom_labels --config_path=configs/abl1-decoder-arch/laom-hopper-vanilla.yaml
```

### 2. I-LAPO-S1 - S1 i-ResNet - S3 MLP

[] Case 2/4:

From here onwards we use `train_laom_labels_inv.py`, not `train_laom_labels.py`, as it is equipped to handle the i-ResNet logic.

```bash
python -m train_laom_labels_inv --config_path=configs/abl1-decoder-arch/s1-inv-hopper-vanilla.yaml
```

### 3. I-LAPO-S3 - S1 Linear - S3 i-ResNet

[] Case 3/4:

```bash
python -m train_laom_labels_inv --config_path=configs/abl1-decoder-arch/s3-inv-hopper-vanilla.yaml
```

### 4. I-LAPO-full - S1 i-ResNet - S3 SAME REUSED i-ResNet

[] Case 4/4:

```bash
python -m train_laom_labels_inv --config_path=configs/abl1-decoder-arch/full-inv-hopper-vanilla.yaml
```

## 4. Ablation 2. Labeled loss coefficient (3 seeds)
- Hopper only
- latent_action_dim=6
- Stay at 125 labeled trajectories (justifies 0.025 choice as 125/5000 = 0.025)
- labeled_loss_coef in {0.01, 0.025}
-> Show if 0.025 helps both methods equally / architectural gain survives regardless of labeled_loss_coef choice
-> Pick best coefficient for I-LAPO* main experiments
-> Eliminates confound variable concerning whether the gain is from the architecture or just from the fact that we are weighting the loss more

### 1. LAOM+supervision - labeled_loss_coef=0.01

[] Case 1/4: NO NEED TO RUN, USE AS BASELINE FROM ABLATION 1 LAOM+supervision S1 Linear - S3 MLP (latent_action_dim=6):

```bash
python -m train_laom_labels --config_path=configs/abl2-labeled-loss-coef/laom-llc0.01-hopper-vanilla.yaml
```

### 2. LAOM+supervision - labeled_loss_coef=0.025

[] Case 2/4:

```bash
python -m train_laom_labels --config_path=configs/abl2-labeled-loss-coef/laom-llc0.025-hopper-vanilla.yaml
```

### 3. I-LAPO* - labeled_loss_coef=0.01

[] Case 3/4:

```bash
python -m train_laom_labels_inv --config_path=configs/abl2-labeled-loss-coef/ilapo-llc0.01-hopper-vanilla.yaml
```

### 4. I-LAPO* - labeled_loss_coef=0.025

[] Case 4/4:

```bash
python -m train_laom_labels_inv --config_path=configs/abl2-labeled-loss-coef/ilapo-llc0.025-hopper-vanilla.yaml
```