import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size


GROUPS = (
    ("Eval", "Eval_"),
    ("Train", "Train_"),
    ("Optimization", None),
)


@dataclass
class LoggedValue:
    value: str
    step: str
    row_number: int


def find_latest_log(default_root: Path) -> Path:
    logs = list(default_root.glob("*/log.csv"))
    if not logs:
        raise FileNotFoundError(
            f"No log.csv files found under {default_root}. Pass a CSV path explicitly."
        )
    return max(logs, key=lambda path: path.stat().st_mtime)


def resolve_log_path(path: Path | None) -> Path:
    if path is None:
        return find_latest_log(Path("exp"))
    if path.is_dir():
        return path / "log.csv"
    return path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not have a CSV header.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} has a header, but no log rows.")

    return reader.fieldnames, rows


def latest_values(fieldnames: list[str], rows: list[dict[str, str]]) -> dict[str, LoggedValue]:
    latest: dict[str, LoggedValue] = {}

    for row_number, row in enumerate(rows, start=2):
        row_step = row.get("step", "").strip()
        for field in fieldnames:
            value = row.get(field, "").strip()
            if value == "":
                continue
            latest[field] = LoggedValue(value=value, step=row_step, row_number=row_number)

    return latest


def format_value(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value

    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def metric_group(metric: str) -> str:
    if metric.startswith("Eval_"):
        return "Eval"
    if metric.startswith("Train_"):
        return "Train"
    return "Optimization"


def ordered_metrics(fieldnames: list[str], latest: dict[str, LoggedValue]) -> list[str]:
    metrics = [field for field in fieldnames if field != "step" and field in latest]
    ordered: list[str] = []

    for group_name, prefix in GROUPS:
        if prefix is None:
            ordered.extend(
                metric
                for metric in metrics
                if metric_group(metric) == group_name
            )
        else:
            ordered.extend(metric for metric in metrics if metric.startswith(prefix))

    ordered.extend(metric for metric in metrics if metric not in ordered)
    return ordered


def print_rule(char: str = "-") -> None:
    width = min(get_terminal_size((100, 20)).columns, 120)
    print(char * width)


def print_table(fieldnames: list[str], latest: dict[str, LoggedValue]) -> None:
    metrics = ordered_metrics(fieldnames, latest)
    if not metrics:
        print("No non-empty metric values found.")
        return

    metric_width = max(len("metric"), *(len(metric) for metric in metrics))
    value_width = max(
        len("value"),
        *(len(format_value(latest[metric].value)) for metric in metrics),
    )
    step_width = max(
        len("logged_at_step"),
        *(len(latest[metric].step or "-") for metric in metrics),
    )

    header = (
        f"{'metric':<{metric_width}}  "
        f"{'value':>{value_width}}  "
        f"{'logged_at_step':>{step_width}}"
    )
    print(header)
    print(f"{'-' * metric_width}  {'-' * value_width}  {'-' * step_width}")

    current_group = None
    for metric in metrics:
        group = metric_group(metric)
        if current_group is not None and group != current_group:
            print()
        current_group = group

        logged = latest[metric]
        print(
            f"{metric:<{metric_width}}  "
            f"{format_value(logged.value):>{value_width}}  "
            f"{(logged.step or '-'):>{step_width}}"
        )


def print_last_log(path: Path, literal_last_row: bool = False) -> None:
    fieldnames, rows = read_rows(path)
    source_rows = rows[-1:] if literal_last_row else rows
    latest = latest_values(fieldnames, source_rows)
    last_step = rows[-1].get("step", "").strip() or "-"

    print_rule("=")
    print(f"Log file : {path}")
    print(f"Rows     : {len(rows)}")
    print(f"Last step: {last_step}")
    if not literal_last_row:
        print("Mode     : latest non-empty value per metric")
    else:
        print("Mode     : literal last CSV row")
    print_rule("=")
    print_table(fieldnames, latest)
    print_rule("=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print the latest values from an HW3 log.csv file."
    )
    parser.add_argument(
        "log_csv",
        nargs="?",
        type=Path,
        help="Path to log.csv. If omitted, the newest exp/*/log.csv is used.",
    )
    parser.add_argument(
        "--literal-last-row",
        action="store_true",
        help="Only print values present in the final CSV row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = resolve_log_path(args.log_csv)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file.")

    print_last_log(path, literal_last_row=args.literal_last_row)


if __name__ == "__main__":
    main()
