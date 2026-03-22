import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_LLM_MODEL_NAME = "qwen_7b"
DEFAULT_LLM_MODEL_PATH = r"E:\HFModels\Qwen2.5-7B-Instruct"


def repo_root():
    return Path(__file__).resolve().parent


def raw_dataset_pickle(root, dataset_name):
    return root / "dataset" / dataset_name / f"{dataset_name}.pkl"


def extracted_dataset_dir(root, dataset_name):
    return root / "dataset" / "extracted_KG" / dataset_name


def extracted_dataset_pickle(root, dataset_name):
    return extracted_dataset_dir(root, dataset_name) / f"{dataset_name}.pkl"


def ensure_extracted_dataset(root, dataset_name, sync_pkl=False):
    source = raw_dataset_pickle(root, dataset_name)
    if not source.exists():
        raise FileNotFoundError(
            f"Raw dataset pickle not found: {source}. "
            f"Expected dataset/{dataset_name}/{dataset_name}.pkl"
        )

    target = extracted_dataset_pickle(root, dataset_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    if sync_pkl or not target.exists():
        shutil.copy2(source, target)

    return target


def _has_graph_files(directory):
    return directory.exists() and any(directory.glob("*.pt"))


def preprocessing_required(root, dataset_name):
    dataset_dir = extracted_dataset_dir(root, dataset_name)
    split_dir = dataset_dir / "split"

    required_split_files = (
        split_dir / "train_indices.txt",
        split_dir / "val_indices.txt",
        split_dir / "test_indices.txt",
    )

    return not (
        extracted_dataset_pickle(root, dataset_name).exists()
        and _has_graph_files(dataset_dir / "graphs" / "claim")
        and _has_graph_files(dataset_dir / "graphs" / "doc")
        and all(path.exists() for path in required_split_files)
    )


def _strip_cli_arg(args, option_name):
    cleaned = []
    i = 0
    while i < len(args):
        current = args[i]
        if current == option_name:
            i += 1
            if i < len(args) and not args[i].startswith("--"):
                i += 1
            continue
        if current.startswith(f"{option_name}="):
            i += 1
            continue
        cleaned.append(current)
        i += 1
    return cleaned


def _has_cli_arg(args, option_name):
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in args)


def build_preprocess_command(root, dataset_name):
    return [sys.executable, str(root / "graph_build.py"), "--data_name", dataset_name]


def build_train_command(root, dataset_name, extra_args):
    cleaned_args = _strip_cli_arg(extra_args, "--train_dataset")
    if not (
        _has_cli_arg(cleaned_args, "--llm_model_name")
        or _has_cli_arg(cleaned_args, "--llm_model_path")
    ):
        cleaned_args = [
            "--llm_model_name",
            DEFAULT_LLM_MODEL_NAME,
            "--llm_model_path",
            DEFAULT_LLM_MODEL_PATH,
            *cleaned_args,
        ]
    return [
        sys.executable,
        str(root / "train.py"),
        "--train_dataset",
        dataset_name,
        *cleaned_args,
    ]


def run_command(command, cwd, dry_run=False):
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare GraphCheck data and launch training."
    )
    parser.add_argument("--train_dataset", type=str, default="MiniCheck_Train")
    parser.add_argument(
        "--sync-pkl",
        action="store_true",
        help="Always copy dataset/<name>/<name>.pkl into dataset/extracted_KG/<name>/.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip graph preprocessing. Fails if required graph artifacts are missing.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild graph artifacts even if they already exist.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare dataset artifacts without launching train.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_known_args(argv)


def main(argv=None):
    args, train_args = parse_args(argv)
    root = repo_root()

    extracted_pkl = ensure_extracted_dataset(
        root, args.train_dataset, sync_pkl=args.sync_pkl
    )
    print(f"Dataset ready: {extracted_pkl}")

    needs_preprocess = args.force_preprocess or preprocessing_required(
        root, args.train_dataset
    )

    if args.skip_preprocess and needs_preprocess:
        raise RuntimeError(
            "Preprocessing was skipped, but graph artifacts are missing. "
            "Run without --skip-preprocess or use --force-preprocess."
        )

    if not args.skip_preprocess and needs_preprocess:
        print(f"Building graph artifacts for {args.train_dataset}...")
        run_command(
            build_preprocess_command(root, args.train_dataset),
            cwd=root,
            dry_run=args.dry_run,
        )
    else:
        print(f"Graph artifacts already available for {args.train_dataset}.")

    if args.prepare_only:
        print(
            "Preparation finished. Training not started because --prepare-only was set."
        )
        return 0

    print(f"Starting training for {args.train_dataset}...")
    run_command(
        build_train_command(root, args.train_dataset, train_args),
        cwd=root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
