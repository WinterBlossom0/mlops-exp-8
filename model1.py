from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

MODEL_PATH = BASE_DIR / "model1.pkl"
METRICS_PATH = BASE_DIR / "metrics.txt"
PLOT_PATH = BASE_DIR / "model_results.png"


def train_and_evaluate(num_runs: int = 10) -> float:
    model = LinearRegression()
    mse_scores: list[float] = []
    plot_data: dict[str, np.ndarray] = {}

    for seed in range(num_runs):
        rng = np.random.RandomState(seed)
        x = 10 * rng.rand(1000).reshape(-1, 1)
        y = 2 * x - 5 + rng.randn(1000).reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=50
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(float(mse))

        plot_data = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "predictions": predictions,
        }

        print(f"Run {seed + 1} MSE: {mse:.4f}")

    average_mse = float(np.mean(mse_scores))
    print(f"Average Mean Squared Error: {average_mse:.4f}")

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        "\n".join(
            [
                f"Average Mean Squared Error = {average_mse:.4f}",
                f"Saved model = {MODEL_PATH.name}",
                f"Runs = {num_runs}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data["x_train"], plot_data["y_train"], color="royalblue", label="Training data")
    plt.scatter(plot_data["x_test"], plot_data["y_test"], color="tomato", label="Testing data")
    plt.scatter(plot_data["x_test"], plot_data["predictions"], color="seagreen", label="Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Training, Testing, and Prediction Results")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()

    return average_mse


if __name__ == "__main__":
    train_and_evaluate()
