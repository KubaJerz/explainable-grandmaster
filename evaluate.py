import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation helper for the hard-coded 5x4 Silverman variant"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print(
        "evaluate.py is intentionally disabled for the 5x4 Silverman rewrite.\n"
        "The previous Stockfish-based evaluator depended on standard 8x8 chess and\n"
        "is not compatible with this custom variant.\n"
        f"Checkpoint provided: {args.checkpoint}"
    )


if __name__ == "__main__":
    main()
