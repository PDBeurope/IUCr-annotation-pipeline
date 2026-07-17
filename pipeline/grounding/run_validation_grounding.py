import argparse
import json

from pipeline.grounding.grounding import term_grounding_with_epmc_json
from pipeline.utils import write_output_json


def main():
    parser = argparse.ArgumentParser(
        description="""
        This script will compute residue grounding and write the results as a gzip-compressed file.
        '"""
    )
    parser.add_argument(
        "--input-json",
        help="Input JSON",
    )
    parser.add_argument(
        "--output-json",
        help="Output JSON",
    )
    parser.add_argument(
        "--validation-dir",
        help="Validation dir",
    )
    parser.add_argument(
        "--sifts-dir",
        help="SIFTS dir",
    )
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        input_json = json.load(f)

    grounded_json = term_grounding_with_epmc_json(
        input_json,
        args.validation_dir,
        args.sifts_dir,
    )

    if args.output_json != "":
        write_output_json(grounded_json, args.output_json)


if __name__ == "__main__":
    main()
