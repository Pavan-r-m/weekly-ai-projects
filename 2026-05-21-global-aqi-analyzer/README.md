# Global Air Quality Index (AQI) Analyzer

A data analysis and visualisation project that **generates a realistic synthetic AQI dataset** for 10 world cities throughout 2023 and produces a comprehensive 5-panel analytical dashboard.

---

## What It Does

The script models daily air quality readings for Delhi, Beijing, Cairo, Lagos, São Paulo, Paris, London, New York, Tokyo, and Sydney — incorporating:

- **Seasonal cycles** — winter PM2.5 peaks in Northern Hemisphere cities
- **Weekly patterns** — ~10 % higher pollution on weekdays (traffic / industry)
- **Long-term trends** — a gradual year-on-year improvement
- **Realistic pollutant correlations** — PM10, NO₂, and CO track PM2.5; O₃ shows an inverse relationship (smog photochemistry)
- **EPA AQI conversion** — raw PM2.5 µg/m³ values are mapped to the standard 0–500 AQI scale with health category labels

---

## Tech Stack & Key Concepts

| Library | Role |
|---------|------|
| **NumPy** | Random number generation, trigonometric seasonal model |
| **Pandas** | Data wrangling, groupBy aggregations, time-series indexing |
| **Matplotlib** | Figure layout (GridSpec), bar charts, box plots, line charts |
| **Seaborn** | Correlation heatmap, colour palettes |

Key concepts: time-series feature engineering, EPA AQI breakpoint interpolation, pollutant correlation analysis, percentage-share stacked charts.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python aqi_analyzer.py
```

**Outputs:**

| File | Description |
|------|-------------|
| `aqi_data.csv` | Raw dataset — 3 650 rows × 8 columns |
| `aqi_analysis.png` | 5-panel composite visualisation |

---

## Example Output (Console)

```
Global AQI Analyzer -- starting up...
  Generating synthetic AQI dataset (365 days x 10 cities)...
  -> 3,650 records generated
  -> Saved raw dataset to aqi_data.csv

==============================================================
   GLOBAL AIR QUALITY INDEX -- ANALYSIS SUMMARY (2023)
==============================================================

  City AQI Statistics:

              Mean  Median  Max  Min
city
Delhi        260.0   257.5  500  101
Beijing      198.5   186.0  500   55
Cairo        161.4   155.0  335   51
Lagos        125.7   120.5  262   51
Sao Paulo     69.6    67.0  175    0
London        50.3    49.0  128    0
Paris         46.2    44.5  120    0
Tokyo         45.1    43.5  117    0
New York      37.3    36.0  103    0
Sydney        23.1    22.0   67    0

  Best air quality :  Sydney  (mean AQI = 23.1)
  Worst air quality:  Delhi   (mean AQI = 260.0)
  City-days with 'Good' AQI:      28.4%
  City-days with 'Hazardous' AQI:  9.3%
```

---

## Visualisation Panels

1. **Monthly AQI Trends** — line chart showing seasonal peaks and city-by-city patterns across 12 months  
2. **AQI Distribution** — colour-coded box plots ordered from worst to best air quality  
3. **Pollutant Correlation Matrix** — heatmap revealing how PM2.5, PM10, NO₂, O₃, and CO relate to each other  
4. **AQI Category Breakdown** — stacked horizontal bars showing the share of "Good / Moderate / Unhealthy / Hazardous" days per city  
5. **Weekday vs Weekend Pattern** — bar chart confirming higher pollution on weekdays across all cities

---

## How It Works

### Data Generation Pipeline

```
CITIES config (base PM2.5, seasonal amplitude, noise)
    ↓
For each city × each day:
    PM2.5 = (base + seasonal_cosine + trend) × weekday_factor + noise
    PM10, NO2, O3, CO = derived via correlated random multipliers
    AQI  = EPA breakpoint interpolation of PM2.5
    ↓
DataFrame of 3 650 rows
```

### AQI Calculation

The EPA uses six concentration breakpoints for PM2.5. Between any two breakpoints, AQI is computed with a linear interpolation:

```
AQI = AQI_low + (PM2.5 - C_low) / (C_high - C_low) × (AQI_high - AQI_low)
```

Breakpoints used:

| PM2.5 (µg/m³) | AQI Range | Category |
|---------------|-----------|----------|
| 0–12          | 0–50      | Good |
| 12.1–35.4     | 51–100    | Moderate |
| 35.5–55.4     | 101–150   | Unhealthy for Sensitive Groups |
| 55.5–150.4    | 151–200   | Unhealthy |
| 150.5–250.4   | 201–300   | Very Unhealthy |
| 250.5–500.4   | 301–500   | Hazardous |

---

## Key Findings

- **Delhi and Beijing** spend significant portions of the year in the "Hazardous" category (AQI > 300), particularly in winter.
- **Sydney and New Zealand cities** consistently achieve "Good" AQI for the majority of the year.
- **O₃ (ozone) is inversely correlated** with PM2.5 — a classic signature of photochemical smog chemistry.
- **Weekday AQI is ~15–20 % higher** than weekend AQI, driven by traffic and industrial activity.
- **Seasonal peaks are pronounced** in South Asian and East Asian cities due to crop burning and winter heating.

---

## No API Key Required

This project uses purely synthetic data generated in-code. No external data downloads or API keys are needed.
