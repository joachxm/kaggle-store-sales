# Store Sales - Time Series Forecasting

Solutions for the Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

**NOTE:** For the interactive plots to display properly, it is necessary to view the notebooks through [nbviewer](https://nbviewer.org/github/joachxm/kaggle-store-sales/tree/main/).


## Notebooks

- [01 - Exploratory data analysis](https://nbviewer.org/github/joachxm/kaggle-store-sales/blob/main/01_exploratory-data-analysis.ipynb)
    - data preparation for the subsequent notebooks
    - analysis of the impact of the covariates

- [02 - Single-key models](https://nbviewer.org/github/joachxm/kaggle-store-sales/blob/main/02_single-key-models.ipynb)
    - build models predicting a single key at a time
    - information about the key is not included in the model inputs
    - score (RMSLE) of 0.426



## Next steps

- **Wide single-key models:** adapt models from part 02 to include the store number and family
- **Full-width models:** build models to predict all keys at once
