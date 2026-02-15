from random import Random


def _stable_seed(payload: dict) -> int:
    # Deterministic seed so identical inputs return identical demo outputs.
    normalized = repr(sorted(payload.items()))
    return sum(ord(ch) for ch in normalized) % (2**32)


def predict_unstable_plaque_and_adverse_outcome(payload: dict) -> dict:
    """Return demo predictions that mimic a model call."""
    rng = Random(_stable_seed(payload))

    unstable_probability = round(rng.uniform(0.05, 0.55), 3)
    adverse_probability = round(
        min(max(0.2 * unstable_probability + rng.uniform(0.02, 0.75), 0.01), 0.99),
        3,
    )

    return {
        "unstable_plaque_probability": unstable_probability,
        "unstable_plaque_prediction": int(unstable_probability >= 0.5),
        "adverse_outcome_probability": adverse_probability,
        "adverse_outcome_prediction": int(adverse_probability >= 0.5),
        "model_version": "demo-random-v1",
    }
