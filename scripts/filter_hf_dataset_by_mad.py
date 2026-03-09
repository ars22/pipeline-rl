import argparse
import statistics

from datasets import get_dataset_config_names, load_dataset


def median_absolute_deviation(scores: list[int | float]) -> float:
    median = statistics.median(scores)
    deviations = [abs(score - median) for score in scores]
    return float(statistics.median(deviations))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a Hugging Face dataset config by score MAD and optionally push it as a new config."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. lm-provers/aops-olympiads")
    parser.add_argument("--source-config", required=True, help="Source dataset config name")
    parser.add_argument("--target-config", required=True, help="Target dataset config name to create/update")
    parser.add_argument("--split", default="train", help="Dataset split to read and push")
    parser.add_argument("--mad-gte", type=float, required=True, help="Keep rows with MAD(scores) >= this value")
    parser.add_argument(
        "--allow-existing-target",
        action="store_true",
        help="Allow pushing even if the target config already exists",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the filtered split to the target config. Without this flag, only print stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    existing_configs = set(get_dataset_config_names(args.repo_id))
    if args.target_config in existing_configs and not args.allow_existing_target:
        raise SystemExit(
            f"Target config {args.target_config!r} already exists in {args.repo_id}. "
            "Pass --allow-existing-target to overwrite it."
        )

    dataset = load_dataset(args.repo_id, args.source_config, split=args.split)
    print(f"Loaded {args.repo_id}/{args.source_config}:{args.split} with {len(dataset)} rows")

    def keep_example(example: dict) -> bool:
        return median_absolute_deviation(example["scores"]) >= args.mad_gte

    filtered = dataset.filter(keep_example, desc=f"Filtering rows with MAD(scores) >= {args.mad_gte}")
    print(
        f"Filtered rows: {len(filtered)} / {len(dataset)} "
        f"({len(filtered) / len(dataset):.1%}) with MAD(scores) >= {args.mad_gte}"
    )

    if not args.push:
        return

    filtered.push_to_hub(
        args.repo_id,
        config_name=args.target_config,
        split=args.split,
    )
    print(
        f"Pushed {args.repo_id}/{args.target_config}:{args.split} "
        f"with {len(filtered)} rows"
    )


if __name__ == "__main__":
    main()
