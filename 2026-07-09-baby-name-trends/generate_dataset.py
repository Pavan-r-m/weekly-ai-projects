"""
generate_dataset.py
--------------------
Builds a realistic, reproducible baby-name popularity dataset covering the
years 1900-2024 for 26 well-known U.S. first names.

Real historical naming data (e.g. from the U.S. Social Security Administration)
is not reachable from this sandbox, so instead of scraping it we MODEL it:
each name is given a real, historically-documented "peak era" (the decade or
two when it was genuinely most popular, based on well-known SSA naming-trend
facts) and a log-normal-shaped popularity curve is generated around that
peak, with year-to-year noise added for realism. A handful of names
(Leslie, Ashley, Jordan, Avery) are modeled as historically documented
"gender-crossover" names whose male/female split shifts over time, which is
a real, well-documented phenomenon in U.S. naming data.

Running this script regenerates `babynames_sample.csv` deterministically
(fixed random seed), so the analysis in `analyze_names.py` is always
reproducible from scratch.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
YEARS = np.arange(1900, 2025)

# Each entry: name -> (peak_year, spread_in_years, base_gender ["F"/"M"/"crossover"],
#                       peak_annual_count, crossover_start/end for names that flip gender)
# Peak years/eras reflect real, well-documented U.S. naming trends.
NAME_PROFILES = {
    "Mary":     (1920, 25, "F", 65000, None),
    "Linda":    (1947, 8,  "F", 99000, None),   # briefly the most popular US girl name ever, 1947
    "Jennifer": (1978, 12, "F", 58000, None),
    "Jessica":  (1990, 8,  "F", 46000, None),
    "Emily":    (2000, 10, "F", 25000, None),
    "Emma":     (2018, 8,  "F", 20000, None),
    "Olivia":   (2020, 6,  "F", 19500, None),
    "Isabella": (2010, 6,  "F", 22500, None),
    "Sophia":   (2013, 6,  "F", 21000, None),
    "Madison":  (2001, 8,  "F", 23000, None),
    "John":     (1915, 25, "M", 90000, None),
    "Michael":  (1975, 20, "M", 70000, None),
    "Robert":   (1940, 20, "M", 80000, None),
    "James":    (1950, 25, "M", 65000, None),
    "William":  (1920, 30, "M", 45000, None),
    "Jacob":    (2001, 8,  "M", 36000, None),
    "Noah":     (2018, 8,  "M", 19500, None),
    "Liam":     (2022, 6,  "M", 20500, None),
    "Ethan":    (2005, 8,  "M", 19000, None),
    # gender-crossover names: (start_decade_share_female, end_decade_share_female)
    "Leslie":   (1955, 20, "crossover", 12000, (0.15, 0.92)),   # male-leaning -> female-dominant
    "Ashley":   (1988, 10, "crossover", 38000, (0.55, 0.98)),   # became overwhelmingly female
    "Jordan":   (1995, 10, "crossover", 15000, (0.05, 0.35)),   # unisex, stayed male-majority
    "Avery":    (2015, 10, "crossover", 11000, (0.20, 0.80)),   # historically male -> now mostly female
    "Riley":    (2005, 12, "crossover", 9500,  (0.35, 0.65)),   # genuinely unisex, near 50/50
    "Casey":    (1985, 15, "crossover", 7000,  (0.30, 0.55)),   # unisex, mild female tilt over time
    "Morgan":   (1998, 12, "crossover", 8500,  (0.25, 0.75)),   # male-leaning -> female-dominant
}


def lognormal_curve(years, peak_year, spread):
    """Return an asymmetric, log-normal-shaped popularity curve peaking at peak_year."""
    x = (years - peak_year) / (spread * 10)
    # log-normal-ish bump: fast rise, slower decay (mirrors real naming-fad shape)
    curve = np.exp(-0.5 * (np.log1p(np.abs(x) * 3) ** 2))
    return curve


def gender_share_for_year(year, start_year, end_year, start_share, end_share):
    """Linearly interpolate female-share of a crossover name between two anchor years."""
    if year <= start_year:
        return start_share
    if year >= end_year:
        return end_share
    frac = (year - start_year) / (end_year - start_year)
    return start_share + frac * (end_share - start_share)


def build_dataset():
    rows = []
    for name, (peak_year, spread, kind, peak_count, crossover) in NAME_PROFILES.items():
        curve = lognormal_curve(YEARS, peak_year, spread)
        curve = curve / curve.max()  # normalize to 1.0 at peak
        noise = RNG.normal(1.0, 0.06, size=len(YEARS))  # +/-6% year-to-year noise
        counts = np.clip(curve * peak_count * noise, 0, None).round().astype(int)

        for year, count in zip(YEARS, counts):
            if count < 5:
                continue  # skip negligible years to keep the dataset realistic-sized
            if kind in ("F", "M"):
                rows.append({"year": int(year), "name": name, "gender": kind, "count": int(count)})
            else:
                start_year, spread2 = peak_year - spread * 5, None
                start_share, end_share = crossover
                # crossover window spans the full observed range of the name
                female_share = gender_share_for_year(
                    year, YEARS.min(), YEARS.max(), start_share, end_share
                )
                female_count = int(round(count * female_share))
                male_count = count - female_count
                if female_count >= 5:
                    rows.append({"year": int(year), "name": name, "gender": "F", "count": female_count})
                if male_count >= 5:
                    rows.append({"year": int(year), "name": name, "gender": "M", "count": male_count})

    df = pd.DataFrame(rows).sort_values(["name", "year", "gender"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv("babynames_sample.csv", index=False)
    print(f"Wrote babynames_sample.csv with {len(dataset):,} rows "
          f"covering {dataset['name'].nunique()} names, "
          f"{dataset['year'].min()}-{dataset['year'].max()}.")
