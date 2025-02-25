Project completed on September 25, 2024.

# Ames Housing Price Prediction

## Project Overview

This project aims to predict housing prices in **Ames, Iowa** using machine learning models. The dataset contains **2051 houses with 81 explanatory variables**, covering a wide range of features such as lot size, neighborhood, number of rooms, and condition of the house.

Two modeling approaches were explored:
	1.	Regularized Linear Regression (Lasso, Ridge, ElasticNet)
	2.	Tree-based Models (Random Forest, XGBoost, CatBoost)

## Repository Contents
* `house_price_EDA.ipynb` – Exploratory Data Analysis (EDA), feature engineering, and model experimentation
* `house_price_model.py` – The main script for training models and generating predictions
* `Report.pdf` – Summary of data preprocessing, modeling techniques, and performance evaluation
* `Instructions.pdf` – Project description and dataset details
* `proj1/` – Directory for datasets or results

## Dataset
The dataset is derived from the Ames Housing dataset, a well-known alternative to the Boston Housing dataset. \
It consists of:
* `train.csv` – 2051 observations with 81 features + target variable (Sale_Price)
* `test.csv` – 879 observations with the same features as train.csv, but without Sale_Price
* `test_y.csv` – Actual Sale_Price values for test data, used only for evaluation

Target Variable: Sale_Price (log-transformed)

## Exploratory Data Analysis (EDA)

The EDA was conducted in `house_price_EDA.ipynb` and includes:

* **Data Cleaning**: Handling missing values, removing inconsistencies
* **Feature Engineering**:
	- One-hot encoding categorical variables
	- Log transformation for skewed variables
	- Standardization of numerical features
* **Outlier Detection**: Using boxplots and winsorization
* **Feature Selection**: Identifying correlated features and eliminating redundancy

## Model Implementation & Performance

Two distinct modeling approaches were used:

1. **Regularized Linear Regression (Lasso, Ridge, ElasticNet)**
* The best performance was achieved using ElasticNet with:
* alpha=0.0001, l1_ratio=0.5, max_iter=10000
* ElasticNet RMSE: 0.1251

2. **Tree-based Models (Random Forest, XGBoost, CatBoost)**
* The best performance was achieved using CatBoost with:
* iterations=1000, bootstrap_type="MVS", max_leaves=64, learning_rate=0.0458, depth=6
* CatBoost RMSE: 0.1191 (best)

## How to Run the Code
1.	Clone the repository:

`git clone https://github.com/SaniyaAbushakimova/Ames-Housing-Price-Prediction.git` \
`cd Ames-Housing-Price-Prediction`

3. Run the model script to train models and generate predictions:
   
`python house_price_model.py`

5. Open the Jupyter Notebook to explore EDA and experiments:
`jupyter notebook house_price_EDA.ipynb`
