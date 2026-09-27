import argparse
import shutil
from pathlib import Path

import kagglehub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    cache_path = Path(kagglehub.dataset_download(args.dataset))
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if cache_path.is_dir():
        shutil.copytree(cache_path, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(cache_path, dest / cache_path.name)

    print(f"Dataset files copied to {dest.resolve()}")


if __name__ == "__main__":
    main()
