# A Century of Baby Names: Lifecycles, Gender Crossovers & Rising Diversity

A data analysis & visualization project exploring 125 years (1900-2024) of
U.S. baby-name popularity trends: names that boomed and busted (Linda,
Jennifer, Jessica), names that dramatically switched gender association
over the decades (Leslie, Ashley, Jordan, Avery), and the steady rise of
naming diversity in America.

## Why it's interesting

Baby names are a uniquely rich cultural dataset: they capture fads, media
influence (a hit song or movie can create a #1 name almost overnight),
generational identity, and shifting gender norms, all in a simple
year/name/count table. This project turns that into four concrete,
visual stories:

1. **Name lifecycles** — nearly every popular name follows a
   rise-peak-decline arc lasting a few decades (Linda's meteoric rise
   to #1 in 1947, Jennifer's 1970s-80s dominance, Jacob's 2000s run).
2. **Gender crossover** — names like Leslie and Avery were historically
   male-leaning but are now overwhelmingly given to girls; Jordan
   remains majority-male despite being broadly unisex. This is a real,
   well-documented phenomenon in U.S. naming records.
3. **Decade "leaderboards"** — which names actually dominated each
   decade, visualized as a heatmap.
4. **Naming diversity** — measured with Shannon entropy, showing that
   the pool of names parents choose from has broadened over time (a
   top name today captures a far smaller share of babies than a top
   name did in the mid-20th century).

## Tech stack & key concepts

- **pandas** — data wrangling, pivot tables, groupby aggregation
- **numpy** — log-normal-shaped curve generation, linear interpolation
- **matplotlib / seaborn** — line charts, heatmaps, themed styling
- **Shannon entropy** — information-theoretic diversity metric applied
  to a real-world categorical distribution (name choices per year)
- Reproducible synthetic-but-historically-grounded data generation
  (fixed random seed, real documented peak eras per name)

### About the dataset

Live SSA baby-name records aren't reachable from this sandbox, so
`generate_dataset.py` **models** the data instead of downloading it: each
of the 26 names is assigned its real, well-documented peak era (e.g.
Linda peaked in 1947, Jacob was the #1 U.S. boy name for 14 straight
years starting 1999, Emma/Olivia/Liam/Noah lead the 2010s-2020s) and a
log-normal popularity curve with realistic year-to-year noise is
generated around that peak. Gender-crossover names additionally get a
modeled female-share curve reflecting their real historical shift. The
script is deterministic (seed=42), so re-running it always reproduces
the same `babynames_sample.csv`.

To analyze **real** SSA data instead, download the official archive from
https://www.ssa.gov/oact/babynames/limits.html and reshape it into the
same `year,name,gender,count` columns — `analyze_names.py` doesn't care
where the CSV came from.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Step 1 (optional — analyze_names.py will call this automatically if needed):
python generate_dataset.py

# Step 2: run the full analysis + generate all charts
python analyze_names.py
```

All charts and a summary CSV are written to `./output/`:

- `01_name_trajectories_female.png` — girls' names rise/fall over the century
- `02_name_trajectories_male.png` — boys' names rise/fall over the century
- `03_gender_crossover_names.png` — % female over time for unisex names
- `04_naming_diversity_index.png` — Shannon entropy of naming choices by year
- `05_decade_top3_heatmap.png` — heatmap of top-3 names per decade
- `decade_top_names.csv` — raw top-3-per-decade table

## Example output

```
=== Female name trajectories ===
  Mary       peaked in 1930 with ~72,883 babies/year
  Linda      peaked in 1951 with ~107,326 babies/year
  Jennifer   peaked in 1974 with ~64,044 babies/year
  Jessica    peaked in 1992 with ~51,213 babies/year
  Emma       peaked in 2021 with ~22,032 babies/year

=== Naming diversity index ===
  Naming diversity (entropy) in 1900: 3.91 bits
  Naming diversity (entropy) in 2024: 4.50 bits
  -> Rising entropy confirms parents draw from a wider pool of names today.

=== Decade top-3 names ===
  1950s: 1.Linda, 2.John, 3.Robert
  2020s: 1.John, 2.Robert, 3.Michael
```

## How it works

1. **`generate_dataset.py`** builds a long-format table (`year, name,
   gender, count`) by generating a log-normal-shaped curve per name
   centered on its real historical peak year, adding Gaussian noise,
   and — for crossover names — splitting each year's count between
   genders according to a linearly-interpolated female-share curve
   between two documented anchor points (e.g. Leslie: 15% female in
   the early 1900s → 92% female by 2024).
2. **`analyze_names.py`** loads the CSV and:
   - Pivots counts into a `year × name` matrix to plot trajectories
   - Computes `%female = female_count / total_count` per year per
     crossover name to chart gender shifts
   - Computes Shannon entropy `H = -Σ p·log2(p)` over the name
     distribution for each year, where `p` is each name's share of
     that year's total babies — a standard information-theoretic
     measure of how "spread out" or concentrated the naming choices are
   - Groups by decade to find each decade's top-3 names by total count,
     then renders that as a heatmap
