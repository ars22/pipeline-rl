import argparse
import statistics

from datasets import get_dataset_config_names, load_dataset


def median_absolute_deviation(scores: list[int | float]) -> float:
    median = statistics.median(scores)
    deviations = [abs(score - median) for score in scores]
    return float(statistics.median(deviations))


def middle_score_fraction(scores: list[int | float]) -> float:
    return sum(2 <= score <= 5 for score in scores) / len(scores)


def extreme_score_fraction(scores: list[int | float]) -> float:
    return sum(score in {0, 7} for score in scores) / len(scores)


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
        "--min-mid-frac",
        type=float,
        default=0.0,
        help="Keep rows with at least this fraction of scores in the middle range [2, 5]",
    )
    parser.add_argument(
        "--max-extreme-frac",
        type=float,
        default=1.0,
        help="Keep rows with at most this fraction of scores at the extremes {0, 7}",
    )
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
        scores = example["scores"]
        return (
            median_absolute_deviation(scores) >= args.mad_gte
            and middle_score_fraction(scores) >= args.min_mid_frac
            and extreme_score_fraction(scores) < args.max_extreme_frac
        )

    filter_parts = [f"MAD(scores) >= {args.mad_gte}"]
    if args.min_mid_frac > 0.0:
        filter_parts.append(f"mid_frac(scores in [2,5]) >= {args.min_mid_frac}")
    if args.max_extreme_frac < 1.0:
        filter_parts.append(f"extreme_frac(scores in {{0,7}}) < {args.max_extreme_frac}")
    filter_desc = " and ".join(filter_parts)

    filtered = dataset.filter(keep_example, desc=f"Filtering rows with {filter_desc}")
    print(
        f"Filtered rows: {len(filtered)} / {len(dataset)} "
        f"({len(filtered) / len(dataset):.1%}) with {filter_desc}"
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
