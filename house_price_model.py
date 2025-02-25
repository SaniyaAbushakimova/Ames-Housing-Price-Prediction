#-------------------------------------------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
import catboost as cb

#-------------------------------------------------------------------------------------------------------------------
# Data Pre-Processing Function
#-------------------------------------------------------------------------------------------------------------------

#Data Pre-Processing Function, Generalized to work with both train & test data
def data_prep(dir, lin_reg = False, tree = False, train_data = False, max_vals = None, dummy_cols = None, encoder = None):
  df = pd.read_csv(dir)
  PID_col = df['PID'].to_numpy()

  # 1. Remove inconsistent rows

  # Only Drop rows in training data
  if train_data:
    # 1.1 Remove: Year_Built > Year_Remod_Add
    df = df[df['Year_Built'] <= df['Year_Remod_Add']]
    # 1.2 Remove: Area = 0 and Materials != NA
    df_idx = df[(df['Mas_Vnr_Area']==0) & (df['Mas_Vnr_Type'].notnull())][['Mas_Vnr_Area', 'Mas_Vnr_Type']].index
    df = df.drop(df_idx, axis=0)

  # 2. Drop low-information and unbalanced columns
  drop_cols = ['PID', 'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating',
                'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area',
                'Longitude','Latitude']

  df = df.drop(columns=drop_cols, axis=1)
  
  # 3. Handle missing data

    # 3.1 Replace Garage_Yr_Blt null values with 0
  df['Garage_Yr_Blt'] = df['Garage_Yr_Blt'].fillna(0)
  

    # 3.2 Replace 'Mas_Vnr_Type' null values with 'No_MasVnr'
  df['Mas_Vnr_Type'] = df['Mas_Vnr_Type'].fillna('No_MasVnr')
  
  # 4. Handle unusual observations: winsorization
  # Make sure you are wensorizing test data based on train's 95th percentile value
  winsorize_cols = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2",
                    "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", "First_Flr_SF",
                    "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF",
                    "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

  # If it is the training data, we want to save max values
  if train_data:                  
    max_vals = {}
  
  for col in winsorize_cols:
    if train_data:
      # Save max values to use in test data preprocessing
      max_val = df[col].quantile(0.95)
      max_vals[col] = max_val
    else:
      max_val = max_vals[col]

    
    df[col] = df[col].clip(upper=max_val)
  
  # 5. Remove highly-correlated columns (LR only)
  if lin_reg:
    # Based on Pearson correlation values explored in EDA
    corr_cols = ['First_Flr_SF', 'TotRms_AbvGrd', 'Garage_Cars']

    df = df.drop(columns=corr_cols, axis=1)

  # 6. Feature Scaling (LR only)
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    # Remove columns that should not be scaled (time related and Y)
    drops = []
    if train_data:
      drops = ["Sale_Price"]
    num_cols = num_cols.drop(drops + ['Year_Built', 'Year_Remod_Add',
                            'Garage_Yr_Blt', 'Year_Sold', 'Mo_Sold'])

    scaler = StandardScaler()

    df[num_cols] = scaler.fit_transform(df[num_cols])

  # 7. Handle categorical variables

    # 7.1 K=2: label encoding
  df['Central_Air'] = df['Central_Air'].map({'Y': 1, 'N': 0})

    # 7.2 K>2: one-hot encoding
  if train_data:
    dummy_cols = df.select_dtypes(include=['object']).columns

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

  if train_data:
    encoded = encoder.fit_transform(df[dummy_cols])
  else:
    encoded = encoder.transform(df[dummy_cols])

  encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(dummy_cols))
  encoded_df = (pd.concat([df.reset_index(drop=True), encoded_df], axis=1).
                      drop(columns=dummy_cols))


  return encoded_df, max_vals, dummy_cols, encoder, PID_col

#-------------------------------------------------------------------------------------------------------------------
# Fitting & Prediction code 
#-------------------------------------------------------------------------------------------------------------------

# Read in & Pre-Process Training Data
linreg_train_data, max_vals_lin, dummy_cols_lin, encoder_lin, _ = data_prep("train.csv", lin_reg=True, train_data=True)
tree_train_data, max_vals_tree, dummy_cols_tree, encoder_tree, _ = data_prep("train.csv", tree=True, train_data=True)

# Pick out y vectors & log scale them
y_linreg = np.log(linreg_train_data['Sale_Price'].to_numpy())
X_linreg = linreg_train_data.drop(columns=['Sale_Price']).to_numpy()

y_tree = np.log(tree_train_data['Sale_Price'].to_numpy())
X_tree = tree_train_data.drop(columns=['Sale_Price']).to_numpy()

# Fit Linear Regression Model with the optimal hyperparameters found prior
linreg_model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000)
linreg_model.fit(X_linreg, y_linreg)

# Fit Tree Based Model with optimal hyperparameters found prior
tree_model = cb.CatBoostRegressor(verbose=0, allow_writing_files=False)
tree_model.fit(X_tree, y_tree)

# Read & Pre-Process Test Data
linreg_test_data, max_vals_lin, dummy_cols_lin, encoder_lin, pid_col = data_prep("test.csv", lin_reg=True, 
                                                                    train_data=False, 
                                                                    max_vals = max_vals_lin, 
                                                                    dummy_cols =dummy_cols_lin,
                                                                    encoder= encoder_lin)
tree_test_data, max_vals_lin, dummy_cols_tree, encoder_tree, pid_col = data_prep("test.csv",  tree=True,  
                                                                  train_data=False, 
                                                                  max_vals = max_vals_tree, 
                                                                  dummy_cols = dummy_cols_tree,
                                                                  encoder=encoder_tree)

# Predict using Linear Regression model & save the exp(log_sale_price) to mysubmission1.txt
lin_reg_pred = linreg_model.predict(linreg_test_data.to_numpy())

df_linreg = pd.DataFrame({'PID': pid_col, 'Sale_Price': np.exp(lin_reg_pred)})
df_linreg.to_csv('mysubmission1.txt', index=False)

# Predict using Tree Based model & save the exp(log_sale_price) to mysubmission2.txt
tree_pred = tree_model.predict(tree_test_data.to_numpy())

df_tree = pd.DataFrame({'PID': pid_col, "Sale_Price": np.exp(tree_pred)})
df_tree.to_csv('mysubmission2.txt', index=False)