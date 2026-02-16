from pathlib import Path

from ml.evaluation.eval import evaluate_adverse_outcome_model


def main() -> None:
    evaluate_adverse_outcome_model(
        features_csv=Path("backend/data/features.csv"),
        model_path=Path("backend/models/adverse_outcome.placeholder.txt"),
    )


if __name__ == "__main__":
    main()
