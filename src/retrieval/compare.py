"""
Print a side-by-side table of all saved experiments.

Usage:
    python -m src.retrieval.compare
"""

import json

from src.config import EXPERIMENTS_DIR


METRICS = ["hit_at_1", "hit_at_5", "mrr_at_10", "hit_at_10", "recall_at_10"]


def main():

    # 1. Read every experiment summary
    files = sorted(EXPERIMENTS_DIR.glob("*_summary.json"))

    if not files:
        print("No experiments found in", EXPERIMENTS_DIR)
        return

    # 2. Print a header
    print(f"{'experiment':<16}{'method':<15}" + "".join(f"{m:>14}" for m in METRICS))
    print("-" * (31 + 14 * len(METRICS)))

    # 3. Print one line per (experiment, method)
    for file in files:

        data = json.loads(file.read_text())
        name = data["settings"]["name"]

        for method, scores in data["summary"].items():
            print(f"{name:<16}{method:<15}"
                  + "".join(f"{scores[m]:>14.3f}" for m in METRICS))

        print()


if __name__ == "__main__":
    main()
