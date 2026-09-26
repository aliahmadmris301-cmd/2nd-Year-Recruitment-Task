# AIML Recruitment 2026 — Ali Ahmad

## 1. Candidate Details
- **Name:** Ali Ahmad
- **Branch / Year:** B.Tech CSE(Data Science)/2-Year
- **Track:** AI-ML Recruitment Task — Second Years

## 2. Problem Statement
Using the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality) — hourly
gas-sensor and reference-analyzer readings from an Italian city, March 2004 to April 2005 — the
goal was to understand historical air-quality patterns and build a model that forecasts a
pollutant's concentration **one hour ahead**, specifically next-hour `CO(GT)`.

## 3. Approach
1. **Cleaned** the raw dataset: fixed the `;` / `,` CSV formatting, merged `Date` + `Time` into an
   hourly datetime index, replaced the `-200` missing-value sentinel with `NaN`, dropped the
   ~90%-missing `NMHC(GT)` column, and filled the remaining gaps with time-based interpolation.
2. **Explored** the data: distributions of key pollutants, weekday/weekend and seasonal patterns,
   a full correlation heatmap, and two specific unusual observations — a sensor-vs-reference
   correlation dip in August, and the single largest pollution spike of the year.
3. **Engineered features** that only ever use information available at prediction time: lag and
   rolling-average versions of `CO(GT)`, plus hour/day/month calendar features (with cyclical
   sin/cos encoding for hour-of-day).
4. **Modelled**: split the data chronologically (80% train / 20% test, no shuffling), then trained
   a Linear Regression and a Random Forest Regressor to predict next-hour `CO(GT)`, benchmarked
   against a naive "no change" persistence baseline.
5. **Analysed** the results: actual-vs-predicted plots, a train-vs-test overfitting check,
   residual/error analysis, feature importance, and a written discussion of time-series data
   leakage and how it was avoided.

## 4. Technologies Used
- **Language:** Python 3
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn
- **Environment:** Jupyter / Google Colab

## 5. Results

| Model | Split | MAE | RMSE | R² |
|---|---|---|---|---|
| Naive persistence (baseline) | test | 0.49 | 0.78 | 0.68 |
| Linear Regression | train | 0.43 | 0.63 | 0.81 |
| Linear Regression | test | 0.44 | 0.65 | 0.78 |
| Random Forest | train | 0.19 | 0.27 | 0.96 |
| Random Forest | test | 0.41 | 0.60 | 0.81 |

Both models clearly beat the naive baseline. Random Forest has the best raw test R² but shows a
much larger train/test gap — i.e. more overfitting — than Linear Regression. The current hour's
own `CO(GT)` reading is by far the most important predictor of the next hour's value.

## 6. Key Learnings
1. Real-world sensor datasets often encode missingness with sentinel values (here, `-200`)
   instead of blanks, and this can produce misleading artifacts — 31 rows initially flagged as
   "duplicates" turned out to just be fully-offline hours, not genuine repeated observations.
2. Building features for a forecasting problem requires care to keep every feature
   **backward-looking only** (e.g. `.shift(k)` applied before `.rolling()`, never a centered
   window), and splitting chronologically rather than randomly, to avoid leaking future
   information into training.
3. A higher test score doesn't automatically mean the "better" model: Random Forest beat Linear
   Regression on test R² but overfit the training data far more, which matters for how much you'd
   trust it under genuinely new, unseen conditions.
4. Correlation and feature-importance analysis can surface real physical phenomena in the data
   (e.g. a metal-oxide sensor's temperature-related cross-sensitivity), not just abstract
   statistics.

## 7. Challenges
**Challenge:** The raw `AirQualityUCI.csv` file uses `;` as a separator and `,` as the decimal
mark (European convention), plus spurious trailing empty columns/rows left over from the original
export — reading it with pandas' default settings silently produces garbage columns and broken
numeric parsing.
**Solution:** Loaded it explicitly with `sep=';', decimal=','`, then dropped fully-empty rows and
columns before any further processing, and cross-checked the result against the dataset's
documented shape (9,357 rows × 15 columns) to confirm the fix worked correctly.
