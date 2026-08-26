"""Fetch the gated dataset from HuggingFace and place it where the code expects.

The data is gated because it derives from BOUQuET and WMT24++, both translation test
sets. See data/README.md for how to request access.

    python -m data.download              # fetch everything
    python -m data.download --only scores instructions
    python -m data.download --list       # show what would be fetched, then exit

Requires `HF_TOKEN` in the environment (or a prior `huggingface-cli login`) and an
accepted licence on the dataset page.
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_ID = "raphaelmerx/purpose-mt"

# group -> (path within the HF snapshot, destination relative to the repo root)
GROUPS: Dict[str, List[tuple]] = {
    "instructions": [
        ("datasets", "datasets"),
    ],
    "annotations": [
        ("annotation_exports", "annotation_exports"),
    ],
    "scores": [
        ("scores", "data/scores"),
    ],
    "controlled": [
        ("translation_results/controlled_mt", "translation_results/controlled_mt"),
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--only", nargs="+", choices=sorted(GROUPS), default=sorted(GROUPS),
        help="Fetch only these groups (default: all).",
    )
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument(
        "--snapshot-dir", type=Path, default=REPO_ROOT / "data" / "gated",
        help="Where to cache the raw snapshot.",
    )
    p.add_argument("--list", action="store_true", help="Show the plan and exit.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace destination files that already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    plan = [(src, dst) for g in args.only for src, dst in GROUPS[g]]
    print(f"Dataset: {args.dataset_id}")
    for src, dst in plan:
        print(f"  {src}  ->  {dst}")
    if args.list:
        return

    if args.dataset_id.startswith("REPLACE_ME/"):
        sys.exit(
            "DATASET_ID is not set. Edit data/download.py (or pass --dataset-id) "
            "with the published dataset name."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: uv sync")

    try:
        snapshot = Path(snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            local_dir=args.snapshot_dir,
            # only pull the groups that were asked for
            allow_patterns=[f"{src}/**" for src, _ in plan],
        ))
    except Exception as exc:  # noqa: BLE001 - surface the cause verbatim
        sys.exit(
            f"Could not download {args.dataset_id}: {exc}\n\n"
            "This dataset is gated. Make sure you have:\n"
            "  1. A HuggingFace account, and HF_TOKEN set (or `huggingface-cli login`)\n"
            "  2. Accepted the terms on the dataset page\n"
            "See https://huggingface.co/docs/hub/en/datasets-gated"
        )

    copied = skipped = 0
    for src_rel, dst_rel in plan:
        src = snapshot / src_rel
        if not src.is_dir():
            print(f"  (missing in snapshot: {src_rel})")
            continue
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            dst = REPO_ROOT / dst_rel / path.relative_to(src)
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied += 1

    print(f"\nCopied {copied} files"
          + (f", skipped {skipped} that already exist (use --overwrite)" if skipped else ""))


if __name__ == "__main__":
    main()
