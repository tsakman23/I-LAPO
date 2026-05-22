"""Generate eval labeled data with a seed that avoids overlap."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py


def collect_original_idxs(h5_path: Path) -> set[int]:
	idxs: set[int] = set()
	with h5py.File(h5_path, "r") as h5_file:
		for key in h5_file:
			idxs.add(int(h5_file[key].attrs["original_traj_idx"]))
	return idxs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run sample_labeled_data with varying seeds until there is no "
			"overlap between train and eval trajectory indices."
		)
	)
	parser.add_argument(
		"--data_path",
		default="/data2/laom/data/hopper-vanilla-5000traj.hdf5",
	)
	parser.add_argument(
		"--save_path",
		default="/data2/laom/data/hopper-vanilla-eval-labeled-125traj.hdf5",
	)
	parser.add_argument(
		"--train_path",
		default="/data2/laom/data/hopper-vanilla-labeled-125traj.hdf5",
	)
	parser.add_argument("--chunk_size", type=int, default=1000)
	parser.add_argument("--num_trajectories", type=int, default=125)
	parser.add_argument("--seed_start", type=int, default=0)
	parser.add_argument("--seed_end", type=int, default=999)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	data_path = Path(args.data_path)
	save_path = Path(args.save_path)
	train_path = Path(args.train_path)

	train_idxs = collect_original_idxs(train_path)

	cmd_base = [
		sys.executable,
		"-m",
		"scripts.sample_labeled_data",
		f"--data_path={data_path}",
		f"--save_path={save_path}",
		f"--chunk_size={args.chunk_size}",
		f"--num_trajectories={args.num_trajectories}",
	]

	for seed in range(args.seed_start, args.seed_end + 1):
		cmd = cmd_base + [f"--seed={seed}"]
		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode != 0:
			print(f"Seed {seed} failed with return code {result.returncode}")
			if result.stdout:
				print(result.stdout)
			if result.stderr:
				print(result.stderr)
			return result.returncode

		eval_idxs = collect_original_idxs(save_path)
		overlap = len(train_idxs & eval_idxs)
		print(f"Seed {seed}: overlap={overlap}")
		if overlap == 0:
			print(f"Found seed with zero overlap: {seed}")
			return 0

	print(
		"No seed found with zero overlap in range "
		f"{args.seed_start}..{args.seed_end}"
	)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
