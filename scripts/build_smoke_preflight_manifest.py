#!/usr/bin/env python3
"""Authenticate the bounded three-cell L40S preflight campaign."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.screen_selection import build_smoke_preflight_manifest_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-result",
        action="append",
        required=True,
        help="Path to one smoke cell's metrics/model_selection.json (repeat 3 times)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    try:
        output.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        parser.error("--output must resolve inside the repository/staging root")
        raise AssertionError("unreachable") from error
    if output.exists():
        parser.error("refusing to overwrite an existing smoke-preflight manifest")

    inputs = []
    for value in args.smoke_result:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        inputs.append(path.resolve())
    document = build_smoke_preflight_manifest_document(inputs, PROJECT_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print(
        "Authenticated smoke campaign "
        f"{document['smoke_campaign_provenance']['campaign_id']} at "
        f"{output.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
