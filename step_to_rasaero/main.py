"""Command-line entry point for generating RASAero files."""

from __future__ import annotations

import argparse

from .aero_csv_parser import parse_rasaero_csv_folder
from .pipeline import generate_rasaero_project


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP-to-RASAero helper")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate extracted geometry and a CDX1 file")
    gen.add_argument("project_dir")
    gen.add_argument("--output-dir")
    gen.add_argument("--open", action="store_true", help="Open the generated CDX1 in RASAero II")

    parse = sub.add_parser("parse-csv", help="Parse RASAero CSV exports")
    parse.add_argument("csv_dir")
    parse.add_argument("--output-dir")
    parse.add_argument("--reference-length-m", type=float, default=1.0)
    parse.add_argument("--cg-from-nose-m", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "generate":
        result = generate_rasaero_project(args.project_dir, args.output_dir, open_after_generate=args.open)
        print(f"CDX1: {result['cdx1_path']}")
        print(f"Geometry: {result['geometry_path']}")
        print(f"Review: {result['review_path']}")
        return 0
    if args.command == "parse-csv":
        result = parse_rasaero_csv_folder(
            args.csv_dir,
            args.output_dir,
            reference_length_m=args.reference_length_m,
            cg_from_nose_m=args.cg_from_nose_m,
        )
        print(f"Aero database: {result['coeff_path']}")
        print(f"Report: {result['report_path']}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
