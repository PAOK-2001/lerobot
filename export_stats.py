import json
import numpy as np


def load_data(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


def filter_outliers(values: np.ndarray, lower_pct: float = 1, upper_pct: float = 99) -> np.ndarray:
    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)
    return values[(values >= lower) & (values <= upper)]


def compute_stats(data: list[dict], key: str) -> tuple[float, float]:
    values = np.array([d[key] for d in data]) * 1000  # convert to ms
    filtered = filter_outliers(values)
    mean = np.mean(filtered)
    sem = np.std(filtered, ddof=1) / np.sqrt(len(filtered))
    return mean, sem


def main(filepath: str):
    data = load_data(filepath)
    metrics = ["obs_fetching", "policy_latency", "obs_to_action"]

    print(f"{'Metric':<20} {'Value (µs)':<20}")
    print("-" * 42)

    for metric in metrics:
        mean, sem = compute_stats(data, metric)
        print(f"{metric:<20} {mean:.4f} ± {sem:.4f}")


if __name__ == "__main__":
    main("latency_timings.json")
