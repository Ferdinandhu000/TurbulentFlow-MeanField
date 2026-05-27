"""
pack_100.py
============
Pick the best checkpoint from 100-epoch runs based on the log file metric.

Usage
-----
# Process all logs in a directory (recommended):
    python pack_100.py --log-dir .logs

# Pass yaml config files:
    python pack_100.py yaml_W_A/01_W_FNO_inside_config.yaml yaml_W_A/05_1_W_CATO-AFNO_inside_config.yaml

# Or pass raw run names (no .yaml suffix):
    python pack_100.py 01_W_FNO_inside_config 05_1_W_CATO-AFNO_inside_config

Logic
-----
For each run name N:
- Read the corresponding log file.
- Find lines like: "Epoch X/100 | Took ... | mean_val_mse: ..."
- Choose the epoch with the smallest metric (configurable).
- Copy the matching checkpoint file from .checkpoints_N/ to:
  checkpoints_best/.checkpoints_N/
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT_DIR = Path(__file__).parent.resolve()
BEST_DIR = ROOT_DIR / "checkpoints_best"

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/100\s+\|\s+Took", re.IGNORECASE)


def _run_name_from_arg(arg: str) -> str:
    p = Path(arg)
    if p.suffix.lower() == ".yaml":
        return p.stem
    return arg


def _normalize_run_name(name: str) -> str:
    name = name.strip().rstrip("/\\")
    if name.startswith(".checkpoints_"):
        return name[len(".checkpoints_") :]
    return name


def _run_name_from_log_file(log_path: Path) -> str:
    fname = log_path.name
    if fname.startswith(".checkpoints_") and fname.endswith(".logs"):
        return fname[len(".checkpoints_") : -len(".logs")]
    if fname.endswith(".logs") or fname.endswith(".log"):
        return log_path.stem
    return fname


def _parse_best_epoch(log_path: Path, metric: str) -> tuple[Optional[int], int]:
    metric_re = re.compile(
        rf"{re.escape(metric)}\s*:\s*([0-9eE.+-]+)",
        re.IGNORECASE,
    )

    best_epoch: Optional[int] = None
    best_value: Optional[float] = None
    total_epoch = 0

    if not log_path.is_file():
        return None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = EPOCH_RE.search(line)
            if not epoch_match:
                continue
            metric_match = metric_re.search(line)
            if not metric_match:
                continue
            epoch = int(epoch_match.group(1))
            value = float(metric_match.group(1))
            if epoch > total_epoch:
                total_epoch = epoch
            if best_value is None or value < best_value:
                best_value = value
                best_epoch = epoch

    return best_epoch, total_epoch


def _find_checkpoint_file(ckpt_dir: Path, epoch: int) -> Optional[Path]:
    pattern = re.compile(rf"(\d+)\.pt$", re.IGNORECASE)
    for p in ckpt_dir.glob("*.pt"):
        m = pattern.search(p.name)
        if m and int(m.group(1)) == epoch:
            return p
    return None


def _checkpoint_dir_for_run(run_name: str) -> Path:
    return ROOT_DIR / f".checkpoints_{_normalize_run_name(run_name)}"


def _dest_dir_for_run(run_name: str) -> Path:
    return BEST_DIR / f".checkpoints_{_normalize_run_name(run_name)}"


def _pack_single(log_path: Path, run_name: str, metric: str) -> None:
    ckpt_dir = _checkpoint_dir_for_run(run_name)
    if not ckpt_dir.is_dir():
        print(f"[SKIP]  checkpoint dir not found: {ckpt_dir}")
        return

    best_epoch, total_epoch = _parse_best_epoch(log_path, metric)
    if best_epoch is None:
        print(f"[SKIP]  no matched epoch lines in: {log_path}")
        return

    ckpt_file = _find_checkpoint_file(ckpt_dir, best_epoch)
    if ckpt_file is None:
        print(f"[SKIP]  no checkpoint for epoch {best_epoch} in {ckpt_dir}")
        return

    dest_dir = _dest_dir_for_run(run_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ckpt_file.name
    shutil.copy2(ckpt_file, dest)
    print(
        f"[OK]    {run_name} | total_epoch={total_epoch} | best_epoch={best_epoch} "
        f"({metric}) -> {dest.relative_to(ROOT_DIR)}"
    )


def pack_best_from_logs(log_paths: Iterable[Path], metric: str) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        run_name = _run_name_from_log_file(log_path)
        _pack_single(log_path, run_name, metric)


def pack_best_from_runs(run_names: Iterable[str], metric: str) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for raw_name in run_names:
        name = _normalize_run_name(raw_name)
        if not name:
            continue
        log_path = ROOT_DIR / f".checkpoints_{name}.logs"
        if not log_path.is_file():
            print(f"[SKIP]  log not found: {log_path}")
            continue
        _pack_single(log_path, name, metric)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick best checkpoint from 100-epoch runs by log metric."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Yaml config paths or run names (without .yaml).",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory containing log files to process in batch (e.g., .logs).",
    )
    parser.add_argument(
        "--metric",
        default="mean_val_mse",
        help="Metric to minimize (default: mean_val_mse).",
    )
    args = parser.parse_args()

    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"Log dir not found: {log_dir}")
            sys.exit(1)
        log_paths = [p for p in log_dir.iterdir() if p.is_file()]
        if not log_paths:
            print(f"No log files found in: {log_dir}")
            sys.exit(1)
        pack_best_from_logs(log_paths, args.metric)
        return

    if not args.runs:
        print("Enter yaml paths or run names (one per line).")
        print("Leave a blank line and press Enter when done.\n")
        runs: list[str] = []
        while True:
            line = input("  Run: ").strip()
            if not line:
                break
            runs.append(line)
    else:
        runs = list(args.runs)

    if not runs:
        print("No runs specified. Exiting.")
        sys.exit(0)

    run_names = [_run_name_from_arg(r) for r in runs]
    pack_best_from_runs(run_names, args.metric)


if __name__ == "__main__":
    main()
