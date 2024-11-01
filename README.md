# Time Series Forecasting

Solutions for the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) Kaggle competition.

**NOTE:** In order for the interactive plots to display properly, it is necessary to view the notebooks through [nbviewer](https://nbviewer.org/github/joachxm/kaggle-store-sales/tree/main/).


## Notebooks

- [01_exploratory_data_analysis.ipynb](https://nbviewer.org/github/joachxm/kaggle-store-sales/blob/main/01_exploratory-data-analysis.ipynb)
    - data preparation for the subsequent notebooks
    - analysis of the impact of the covariates

- [02_single_key_models.ipynb](https://nbviewer.org/github/joachxm/kaggle-store-sales/blob/main/02_single-key-models.ipynb)
    - build models predicting a single key at a time
    - information about the key is not included in the model inputs
    - score (RMSLE) of 0.426



## Next steps

- **Wide single-key models:** adapt models from part 02 to include the store number and family
- **Full-width models:** build models to predict all keys at once
