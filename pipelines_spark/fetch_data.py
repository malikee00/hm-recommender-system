import argparse
import os


REQUIRED = ["customers.csv", "articles.csv", "transactions_train.csv"]


def main():
    parser = argparse.ArgumentParser(description="Phase 2.1 - fetch/check raw files exist")
    parser.add_argument("--raw-dir", type=str, default="data/raw/hm/reference")
    args = parser.parse_args()

    missing = []
    for f in REQUIRED:
        p = os.path.join(args.raw_dir, f)
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "Missing required raw files:\n" + "\n".join(missing) +
            "\n\nPlace Kaggle CSVs into: data/raw/hm/reference/"
        )

    print("[DONE] Raw files present:")
    for f in REQUIRED:
        print(" -", os.path.join(args.raw_dir, f))


if __name__ == "__main__":
    main()
