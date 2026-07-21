"""
predict.py
-----------
Small CLI utility that loads the trained model (outputs/model.joblib) and
predicts compressive strength for one or more custom concrete mix designs.

Example:
    python train_model.py          # trains + saves outputs/model.joblib
    python predict.py --cement 350 --slag 100 --fly_ash 0 --water 175 \\
        --superplasticizer 5 --coarse_aggregate 1000 --fine_aggregate 750 --age 28

If no arguments are given, predicts strength for three example mixes
(low-strength, standard, and high-performance) so the script is runnable
with zero configuration.
"""

import argparse
import os
import joblib
import numpy as np

FEATURE_COLS = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
]

EXAMPLE_MIXES = {
    "Low-strength mix (high water-cement ratio, 7-day cure)": [
        200, 0, 0, 200, 0, 1000, 800, 7
    ],
    "Standard structural mix (28-day cure)": [
        350, 50, 0, 180, 6, 1000, 750, 28
    ],
    "High-performance mix (low water, slag+fly ash, 90-day cure)": [
        450, 120, 80, 150, 15, 1000, 700, 90
    ],
}


def load_model():
    model_path = os.path.join("outputs", "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found at outputs/model.joblib. "
            "Run `python train_model.py` first."
        )
    return joblib.load(model_path)


def predict_one(model, cement, slag, fly_ash, water, superplasticizer,
                 coarse_aggregate, fine_aggregate, age) -> float:
    x = np.array([[cement, slag, fly_ash, water, superplasticizer,
                    coarse_aggregate, fine_aggregate, age]])
    return float(model.predict(x)[0])


def main():
    parser = argparse.ArgumentParser(description="Predict concrete compressive strength.")
    parser.add_argument("--cement", type=float)
    parser.add_argument("--slag", type=float, dest="blast_furnace_slag")
    parser.add_argument("--fly_ash", type=float)
    parser.add_argument("--water", type=float)
    parser.add_argument("--superplasticizer", type=float)
    parser.add_argument("--coarse_aggregate", type=float)
    parser.add_argument("--fine_aggregate", type=float)
    parser.add_argument("--age", type=float, help="curing age in days")
    args = parser.parse_args()

    model = load_model()

    provided = [
        args.cement, args.blast_furnace_slag, args.fly_ash, args.water,
        args.superplasticizer, args.coarse_aggregate, args.fine_aggregate, args.age,
    ]

    if all(v is not None for v in provided):
        strength = predict_one(model, *provided)
        print(f"Predicted compressive strength: {strength:.2f} MPa")
    else:
        print("No full mix specified via CLI args — running example mixes:\n")
        for label, values in EXAMPLE_MIXES.items():
            strength = predict_one(model, *values)
            mix_str = ", ".join(f"{k}={v}" for k, v in zip(FEATURE_COLS, values))
            print(f"{label}")
            print(f"  Mix: {mix_str}")
            print(f"  Predicted compressive strength: {strength:.2f} MPa\n")


if __name__ == "__main__":
    main()
