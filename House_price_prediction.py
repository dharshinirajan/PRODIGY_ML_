import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pip install -q pycaret >/dev/null 2>&1 # library used for model trainning and evaluation
pip install -q scipy
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
sns.set_palette("pastel")
sns.set_style('darkgrid')
warnings.filterwarnings("ignore")
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')
target=train_df.SalePrice
df = pd.concat([train_df.drop('SalePrice', axis=1), test_df])
df.head()
df.info()
df.MSSubClass=df.MSSubClass.astype('str')
df.drop('Id',axis=1,inplace=True)
df.describe()
describe=df.describe().T
describe['nunique']=df.nunique()
describe['NULLS']=df.isna().sum()
describe
categorical_1=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType'
   ,'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
for column in categorical_1:
    df[column] = df[column].fillna("None")categorical_2=['MasVnrType','MSZoning','Functional','Utilities','SaleType','Exterior2nd','Exterior1st',
         'Electrical' ,'KitchenQual']
for column in categorical_1:
    df[column] = df[column].fillna(df[column].mode()[0])
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def optimize_knn_imputer(data, col, neighbors_list):
    numerical_data = data.select_dtypes(exclude='O')
    clean_numerical_cols = numerical_data.isna().sum()[numerical_data.isna().sum()==0].index

    X_train = numerical_data[clean_numerical_cols][numerical_data[col].isna()==0]
    y_train = numerical_data[col][numerical_data[col].isna()==0]

    X_test = numerical_data[clean_numerical_cols][numerical_data[col].isna()==1]

    param_grid = {'n_neighbors': neighbors_list}

    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    best_n_neighbors = grid_search.best_params_['n_neighbors']
    best_knn = KNeighborsRegressor(n_neighbors=best_n_neighbors)
    best_knn.fit(X_train, y_train)

    y_pred = best_knn.predict(X_test)

    data[col][data[col].isna()==1] = y_pred

    return data
num_f = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
    'BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']

clean_df = df.copy()

for col in num_f:
    clean_df = optimize_knn_imputer(clean_df, col, neighbors_list=[1, 3, 5, 7, 9])
  clean_df['TotalArea']=clean_df['LotFrontage']+clean_df['LotArea']

clean_df['Total_Home_Quality'] = clean_df['OverallQual'] + clean_df['OverallCond']

clean_df['Total_Bathrooms'] = (clean_df['FullBath'] + (0.5 * clean_df['HalfBath']) +
                               clean_df['BsmtFullBath'] + (0.5 * clean_df['BsmtHalfBath']))
clean_df["AllSF"] = clean_df["GrLivArea"] + clean_df["TotalBsmtSF"]

clean_df["AvgSqFtPerRoom"] = clean_df["GrLivArea"] / (clean_df["TotRmsAbvGrd"] +
                                                       clean_df["FullBath"] +
                                                       clean_df["HalfBath"] +
                                                       clean_df["KitchenAbvGr"])

clean_df["totalFlrSF"] = clean_df["1stFlrSF"] + clean_df["2ndFlrSF"]
