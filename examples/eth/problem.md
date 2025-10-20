# Electricity Transformer Temperature Forecasting Challenge (ETT-m1 Subset)

## Overview
This competition challenges participants to develop cutting-edge solutions for **long sequence time-series forecasting** using the **ETT-small-m1** dataset. The fundamental goal is to accurately predict the future **Oil Temperature (OT)** of a single electricity transformer (Region 1) at a **minute-level resolution**. Precise prediction is critical for electrical transformer safety and for optimizing electricity usage, preventing waste and equipment depreciation.

---

## ETTm1 Dataset Details
The ETT-small-m1 dataset provides two years of real-world multivariate time-series data, collected from a single region of a province in China (ETT-m1). Each data point is recorded every **minute**, resulting in approximately 70,080 data points for the two-year period. The data exhibits complex patterns including short-term daily, long-term weekly/seasonal periodicities, and overall long-term trends.

| Metric | Detail |
| :--- | :--- |
| **Objective** | Multivariate long-sequence time-series forecasting. |
| **Time Frame** | July 2016 to July 2018 (2 years). |
| **Data Granularity** | Minute-level (4 data points per hour, i.e., 15-minute intervals). |

---

## Feature Description
Each data point consists of 8 features. The **Oil Temperature (OT)** is the target variable, with six external power load features serving as auxiliary predictors.

| Field | Description | Role |
| :----: | :----: | :----: |
| `date` | The recorded date and time stamp. | Temporal/Index |
| **`OT`** | **Oil Temperature** (The final variable to be predicted). | **Target** |
| `HUFL` | High UseFul Load. | Exogenous |
| `HULL` | High UseLess Load. | Exogenous |
| `MUFL` | Middle UseFul Load. | Exogenous |
| `MULL` | Middle UseLess Load. | Exogenous |
| `LUFL` | Low UseFul Load. | Exogenous |
| `LULL` | Low UseLess Load. | Exogenous |
