#!/usr/bin/env python3
"""Build the only lock eligible to freeze a validation-screen winner."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.screen_selection import build_selected_method_lock_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete 15-cell screen and write its deterministic "
            "selected-method lock"
        )
    )
    parser.add_argument(
        "--screen-result",
        action="append",
        required=True,
        help="Path to one cell's metrics/model_selection.json (repeat 15 times)",
    )
    parser.add_argument("--output", required=True, help="Lock JSON to create")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()
    try:
        output_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        parser.error("--output must resolve inside the repository/staging root")
        raise AssertionError("unreachable") from error
    if output_path.exists():
        parser.error("refusing to overwrite an existing selected-method lock")

    result_paths = []
    for value in args.screen_result:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        result_paths.append(path.resolve())
    lock_document = build_selected_method_lock_document(
        result_paths, PROJECT_ROOT
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(lock_document, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(
        f"Wrote selected method {lock_document['selected_method']} lock to "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
