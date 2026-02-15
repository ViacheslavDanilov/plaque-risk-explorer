from pathlib import Path

from ml.preprocessing.build_features import build_features


def main() -> None:
    build_features(
        input_csv=Path("backend/data/source.csv"),
        output_csv=Path("backend/data/features.generated.csv"),
    )


if __name__ == "__main__":
    main()
