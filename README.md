# Predicting HDB Resale Flat Prices in Singapore

**Authors:** Neeraj Lakshmanan, Sanchita Shrivastava, Jack Wang  
**Course:** CSE 160 — Introduction to Data Science  
**Instructor:** Professor Brian Davison, Lehigh University

---

## Overview

This project investigates whether historical transaction data can accurately predict resale flat prices in Singapore's public housing market (HDB). Using data from the Housing & Development Board covering January 2017 to the March 2026, we built and compared three regression models — Linear Regression, Random Forest, and XGBoost — following the CRISP-DM framework.

The best model (XGBoost) achieved an **R² of 0.978** and a **Mean Absolute Error of SGD $19,537.76**, meaning predictions are off by only ~3.7% of the average transaction price on average.

---

## Dataset

**Source:** Housing & Development Board (HDB)  
**File:** `ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv`  
**Citation:** Housing & Development Board. (2021). *Resale flat prices based on registration date from Jan-2017 onwards* (2026) [Dataset]. data.gov.sg. Retrieved March 26, 2026 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view

| Property | Value |
|---|---|
| Rows | 227,735 |
| Columns | 11 |
| Missing values | None |
| Coverage | Jan 2017 – Mar 2026 |

**Columns:**
- `month` — transaction month (date)
- `town` — HDB town (categorical)
- `flat_type` — flat type, e.g. 3 ROOM, 4 ROOM (categorical)
- `block` — block number (categorical)
- `street_name` — street name (categorical)
- `storey_range` — storey range, e.g. 04 TO 06 (categorical)
- `floor_area_sqm` — floor area in square meters (numerical)
- `flat_model` — flat model, e.g. Improved, New Generation (categorical)
- `lease_commence_date` — year lease commenced (numerical)
- `remaining_lease` — remaining lease duration (categorical → converted to months)
- `resale_price` — **target variable**, resale price in SGD (numerical)

---

## Project Structure

```
cse160finalproject/
├── ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv   # raw dataset
├── housingdatageneralinsights.Rmd     # R EDA: box plots, price trends by storey & flat type
├── housingdatalinreg.Rmd              # R: Linear Regression baseline model
├── housingdatarandomforest.Rmd        # R: Random Forest ensemble model
├── housingdataxgboost.Rmd             # R: XGBoost Regressor model
└── CSE160 Project Description - Neeraj, Jack & Sanchita.pdf  # full project report
```

---

## Methodology

### Feature Engineering

The following transformations were applied to the raw data before modeling:

- **Label encoding** — `town`, `flat_type`, `storey_range`, `flat_model` converted to integers
- **Remaining lease** — parsed from text format (e.g. "61 years 04 months") into total months
- **`time_index`** — continuous count of months elapsed since the earliest date in the dataset; captures macroeconomic trends, inflation, and shifting market demand
- **`month_num`** — calendar month number extracted to test for seasonality
- **`street_avg_price`** — average resale price per street, calculated **only on training data** to prevent data leakage

### Features Used in All Models

```
town, flat_type, floor_area_sqm, storey_range, flat_model,
remaining_lease, time_index, month_num, street_avg_price
```

### Train / Test Split

80/20 split — ~182,000 training rows, ~45,547 test rows (fixed `set.seed(42)`)

---

## Models & Results

### 1. Linear Regression (Baseline)

| Metric | Score |
|---|---|
| R² | 0.880 |
| MAE | SGD $49,354.32 |

5-fold cross-validation was used.

### 2. Random Forest

| Metric | Score |
|---|---|
| R² | 0.951 |
| MAE | SGD $27,555.22 |

Trained on a random subsample of 50,000 rows due to computational constraints; evaluated on the full 45,547-row held-out test set. Hyperparameter `mtry` was tuned via 5-fold CV across values {3, 6, 9} with 100 trees — `mtry = 6` was selected (lowest RMSE). Feature importance measured by Mean Decrease in Impurity (MDI).

### 3. XGBoost Regressor (Best Model)

| Metric | Score |
|---|---|
| R² | 0.978 |
| MAE | SGD $19,537.76 |
| MAE as % of mean price | ~3.7% |

Trained on the full training set. Key hyperparameters: `eta = 0.1`, `max_depth = 6`, `nrounds = 3000` with early stopping at 20 rounds. Feature importance measured by Information Gain.

---

## Key Insights

**Insight 1 — Floor area is the primary price driver.**  
While the baseline linear model struggled to capture this relationship clearly, both ensemble models confirmed floor area as the top predictor via feature importance.

**Insight 2 — Street matters more than town.**  
`street_avg_price` consistently outperformed `town` as a predictor. Buyers should research at the street level rather than relying on a town's general reputation.

**Insight 3 — Seasonality is nearly useless.**  
`month_num` had the lowest impact of all features (R² decrease of only 0.0001 in Linear Regression). Trying to time the market by month provides almost no advantage — property value is driven by size, location, and broader macroeconomic trends.

---

## Setup & Requirements

### R (for modeling notebooks)

Install the following R packages:

```r
install.packages(c("ggplot2", "lubridate", "dplyr", "scales", "caret", "randomForest", "xgboost"))
```

### Running the notebooks

Open any `.Rmd` file in RStudio and click **Run All** or **Knit to HTML**. All notebooks load the dataset from the same directory — make sure the CSV file is present.

---

## Dataset Credit

Housing & Development Board. (2021). *Resale flat prices based on registration date from Jan-2017 onwards* (2026) [Dataset]. data.gov.sg. Retrieved March 26, 2026 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
