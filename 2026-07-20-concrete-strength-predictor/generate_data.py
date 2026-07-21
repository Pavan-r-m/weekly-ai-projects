"""
generate_data.py
-----------------
Generates a synthetic but physically-plausible concrete compressive strength
dataset, modeled after the well-known UCI "Concrete Compressive Strength"
dataset (Yeh, 1998). Real concrete strength depends on the mix of cement,
water, aggregates, admixtures, and curing age. We simulate this relationship
with a domain-informed formula (based on a simplified Abrams' water-cement
ratio law plus contributions from supplementary cementitious materials and
a log-curing-age term) and add realistic Gaussian noise, so the resulting
dataset behaves like real lab data without needing an internet download.

Run this script standalone to regenerate data/concrete_data.csv, or import
`generate_dataset()` from other scripts (train_model.py imports it directly).
"""

import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 1030, random_seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic concrete mix dataset.

    Columns (mirrors the real UCI concrete dataset's feature names, all in
    kg per cubic meter of mixture, except age in days and strength in MPa):
        cement, blast_furnace_slag, fly_ash, water, superplasticizer,
        coarse_aggregate, fine_aggregate, age, compressive_strength
    """
    rng = np.random.default_rng(random_seed)

    # --- Simulate realistic mix design ranges (based on typical concrete mixes) ---
    cement = rng.uniform(100, 540, n_samples)
    blast_furnace_slag = rng.uniform(0, 360, n_samples)
    fly_ash = rng.uniform(0, 200, n_samples)
    water = rng.uniform(120, 250, n_samples)
    superplasticizer = rng.uniform(0, 32, n_samples)
    coarse_aggregate = rng.uniform(800, 1150, n_samples)
    fine_aggregate = rng.uniform(590, 995, n_samples)
    age = rng.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], size=n_samples)

    # --- Domain-informed strength formula ---
    # 1. Water-to-cementitious-material ratio drives strength (lower ratio -> stronger),
    #    following a simplified Abrams'-law-style exponential relationship.
    cementitious = cement + 0.5 * blast_furnace_slag + 0.6 * fly_ash + 1e-6
    w_c_ratio = water / cementitious
    base_strength = 105 / (1 + 4.5 * w_c_ratio)

    # 2. Superplasticizer improves workability, allowing lower water content
    #    for a given strength -> small positive bonus.
    sp_bonus = 0.35 * superplasticizer

    # 3. Curing age: strength gains follow a logarithmic curve that levels off.
    age_factor = np.log1p(age) / np.log1p(28)  # normalized so 28 days = factor of 1
    age_bonus = base_strength * (age_factor - 1) * 0.6

    # 4. Slag/fly ash contribute slowly over time (pozzolanic reaction).
    scm_bonus = 0.02 * blast_furnace_slag * np.minimum(age_factor, 1.5) + \
                0.015 * fly_ash * np.minimum(age_factor, 1.5)

    # 5. Aggregate ratio has a mild effect on strength (too much fine aggregate weakens mix).
    agg_ratio = fine_aggregate / (coarse_aggregate + 1e-6)
    agg_penalty = -6.0 * np.clip(agg_ratio - 0.75, 0, None)

    strength = base_strength + sp_bonus + age_bonus + scm_bonus + agg_penalty

    # Add measurement/lab noise (real concrete testing has meaningful variance)
    noise = rng.normal(0, 3.0, n_samples)
    strength = strength + noise

    # Clip to physically plausible bounds (MPa)
    strength = np.clip(strength, 2, 90)

    df = pd.DataFrame({
        "cement": cement.round(2),
        "blast_furnace_slag": blast_furnace_slag.round(2),
        "fly_ash": fly_ash.round(2),
        "water": water.round(2),
        "superplasticizer": superplasticizer.round(2),
        "coarse_aggregate": coarse_aggregate.round(2),
        "fine_aggregate": fine_aggregate.round(2),
        "age": age.astype(int),
        "compressive_strength": strength.round(2),
    })

    return df


if __name__ == "__main__":
    import os

    dataset = generate_dataset()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "concrete_data.csv")
    dataset.to_csv(out_path, index=False)
    print(f"Generated {len(dataset)} rows -> {out_path}")
    print(dataset.describe().round(2))
