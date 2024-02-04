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
clean_df['MoSold'] = (-np.cos(0.5236 * clean_df['MoSold']))
def Gar_category(cat):
    if cat <= 250:
        return 1
    elif cat <= 500 and cat > 250:
        return 2
    elif cat <= 1000 and cat > 500:
        return 3
    return 4
clean_df['GarageArea_cat'] = clean_df['GarageArea'].apply(Gar_category).astype('str')

def Low_category(cat):
    if cat <= 1000:
        return 1
    elif cat <= 2000 and cat > 1000:
        return 2
    elif cat <= 3000 and cat > 2000:
        return 3
    return 4
clean_df['GrLivArea_cat'] = clean_df['GrLivArea'].apply(Low_category).astype('str')

def fl1_category(cat):
    if cat <= 500:
        return 1
    elif cat <= 1000 and cat > 500:
        return 2
    elif cat <= 1500 and cat > 1000:
        return 3
    elif cat <= 2000 and cat > 1500:
        return 4
    return 5
clean_df['1stFlrSF_cat'] = clean_df['1stFlrSF'].apply(fl1_category).astype('str')
clean_df['2ndFlrSF_cat'] = clean_df['2ndFlrSF'].apply(fl1_category).astype('str')

def bsmtt_category(cat):
    if cat <= 500:
        return 1
    elif cat <= 1000 and cat > 500:
        return 2
    elif cat <= 1500 and cat > 1000:
        return 3
    elif cat <= 2000 and cat > 1500:
        return 4
    return 5
clean_df['TotalBsmtSF_cat'] = clean_df['TotalBsmtSF'].apply(bsmtt_category).astype('str')

def bsmt_category(cat):
    if cat <= 500:
        return 1
    elif cat <= 1000 and cat > 500:
        return 2
    elif cat <= 1500 and cat > 1000:
        return 3
    elif cat <= 2000 and cat > 1500:
        return 4
    return 5
clean_df['BsmtUnfSF_cat'] = clean_df['BsmtUnfSF'].apply(bsmt_category).astype('str')

def lot_category(cat):
    if cat <= 50:
        return 1
    elif cat <= 100 and cat > 50:
        return 2
    elif cat <= 150 and cat > 100:
        return 3
    return 4
clean_df['LotFrontage_cat'] = clean_df['LotFrontage'].apply(lot_category).astype('str')

def lot_category1(cat):
    if cat <= 5000:
        return 1
    elif cat <= 10000 and cat > 5000:
        return 2
    elif cat <= 15000 and cat > 10000:
        return 3
    elif cat <= 20000 and cat > 15000:
        return 4
    elif cat <= 25000 and cat > 20000:
        return 5
    return 6
clean_df['LotArea_cat'] = clean_df['LotArea'].apply(lot_category1).astype('str')

def year_category(yb):
    if yb <= 1910:
        return 1
    elif yb <= 1950 and yb > 1910:
        return 2
    elif yb >= 1950 and yb < 1980:
        return 3
    elif yb >= 1980 and yb < 2000:
        return 4
    return 5



clean_df['YearBuilt_cat'] = clean_df['YearBuilt'].apply(year_category).astype('str').astype('str').astype('str')
clean_df['YearRemodAdd_cat'] = clean_df['YearRemodAdd'].apply(year_category).astype('str').astype('str')
clean_df['GarageYrBlt_cat'] = clean_df['GarageYrBlt'].apply(year_category).astype('str')

def vnr_category(cat):
    if cat <= 250:
        return 1
    elif cat <= 500 and cat > 250:
        return 2
    elif cat <= 750 and cat > 500:
        return 3
    return 4

clean_df['MasVnrArea_cat'] = clean_df['MasVnrArea'].apply(vnr_category).astype('str')

def allsf_category(yb):
    if yb <= 1000:
        return 1
    elif yb <= 2000 and yb > 1000:
        return 2
    elif yb >= 3000 and yb < 2000:
        return 3
    elif yb >= 4000 and yb < 3000:
        return 4
    elif yb >= 5000 and yb < 4000:
        return 5
    elif yb >= 6000 and yb < 5000:
        return 6
    return 7

clean_df['AllSF_cat'] = clean_df['AllSF'].apply(allsf_category).astype('str')
df1=clean_df.copy()
numerical_features = df1.select_dtypes(include=np.number).columns

# Set up subplots
fig, axes = plt.subplots(nrows=len(numerical_features), ncols=3, figsize=(15, 5 * len(numerical_features)))
fig.subplots_adjust(hspace=0.5)

# Loop through each numerical feature
for i, feature in enumerate(numerical_features):
    # Original Distribution
    sns.distplot(df1[feature], kde=True, ax=axes[i, 0],fit=scipy.stats.norm)
    axes[i, 0].set_title(f'Original Distribution - {feature}')

    # Log-Transformed Distribution
    log_transformed = np.log1p(df1[feature])  # Adding 1 to avoid log(0)
    sns.distplot(log_transformed, kde=True, ax=axes[i, 1],fit=scipy.stats.norm)
    axes[i, 1].set_title(f'Log-Transformed Distribution - {feature}')

    # Square Root Transformed Distribution
    sqrt_transformed = np.sqrt(df1[feature])
    sns.distplot(sqrt_transformed, kde=True, ax=axes[i, 2],fit=scipy.stats.norm)
    axes[i, 2].set_title(f'Square Root Transformed Distribution - {feature}')

plt.show()
skew_df = pd.DataFrame(df1.select_dtypes(np.number).columns, columns=['Feature'])
skew_df['Original_Skewness'] = skew_df['Feature'].apply(lambda feature: scipy.stats.skew(df1[feature]))
skew_df['original_ kurtosis'] = skew_df['Feature'].apply(lambda feature: scipy.stats.kurtosis(df1[feature]))

skew_df['log_transformed_Skewness'] = skew_df['Feature'].apply(lambda feature: scipy.stats.skew(  np.log1p(df1[feature]) )  )
skew_df['log_transformed_ kurtosis'] = skew_df['Feature'].apply(lambda feature: scipy.stats.kurtosis(  np.log1p(df1[feature]) )  )

skew_df['sqrt_transformed_Skewness'] = skew_df['Feature'].apply(lambda feature: scipy.stats.skew(  np.sqrt(df1[feature]) )  )
skew_df['sqrt_transformed_ kurtosis'] = skew_df['Feature'].apply(lambda feature: scipy.stats.kurtosis(  np.sqrt(df1[feature]) )  )
skew_df.set_index('Feature', inplace=True)
skew_df
to_log1p=['LotArea','MasVnrArea','BsmtFinSF2','1stFlrSF','LowQualFinSF','GrLivArea','KitchenAbvGr','TotRmsAbvGrd'
         ,'GarageYrBlt','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal'
         ,'TotalArea','AllSF','AvgSqFtPerRoom','totalFlrSF']
to_sqrt=['LotFrontage','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath','Total_Bathrooms',]
for col in to_log1p:
    df1[col]=np.log1p (df1[col])
for col in to_sqrt:
    df1[col]=np.sqrt (df1[col])
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5 ))
feature='saleprice'
# Original Distribution
sns.distplot(target, kde=True, ax=axes[0],fit=scipy.stats.norm)
axes[0].set_title(f'Original Distribution - {feature}')


    # Log-Transformed Distribution
log_transformed = np.log1p(target)  # Adding 1 to avoid log(0)
sns.distplot(log_transformed, kde=True, ax=axes[1],fit=scipy.stats.norm)
axes[1].set_title(f'Log-Transformed Distribution - {feature}')

    # Square Root Transformed Distribution
sqrt_transformed = np.sqrt(target)
sns.distplot(sqrt_transformed, kde=True, ax=axes[2],fit=scipy.stats.norm)
axes[2].set_title(f'Square Root Transformed Distribution - {feature}')

plt.show()
print(f'original skew = {scipy.stats.skew(target)} \noriginal kurtosis = {scipy.stats.kurtosis(target)}')
print(f'log skew = { scipy.stats.skew(np.log1p (target) )} \nlog kurtosis = {scipy.stats.kurtosis(np.log1p (target))}')
print(f'sqrt skew = { scipy.stats.skew(np.sqrt (target) )} \nsqrt kurtosis = {scipy.stats.kurtosis(np.sqrt (target))}')
log_target=np.log1p(target)
encoded_df=df1.copy()
encoded_df=pd.get_dummies(df1)
encoded_df.info()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
scaled_df=pd.DataFrame(robust_scaler.fit_transform(encoded_df), index=encoded_df.index, columns=encoded_df.columns)
scaled_df.head()
X_train=scaled_df.iloc[:1460]
y_train=log_target
X_test=scaled_df.iloc[1460:]
!pip install catboost
from pycaret.regression import setup, compare_models
from pycaret.regression import *
from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge, OrthogonalMatchingPursuit
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
pycaret_setup = setup(data=pd.concat([X_train, y_train], axis=1), target='SalePrice',session_id=1183)
compare_models(['catboost','br','omp','ridge','gbr'])
catboost,br,omp,ridge,gbr=create_model('catboost',verbose=False),create_model('br',verbose=False),create_model('omp',verbose=False),create_model('ridge',verbose=False),create_model('gbr',verbose=False)
catboost.fit(X_train,y_train,verbose=False)
catboost_prediction=np.exp(catboost.predict(X_test))
catboost_submission = pd.concat([test_df['Id'], pd.Series(catboost_prediction, name='SalePrice')], axis=1)
catboost_submission.to_csv('cat_boost_submission_try1.csv', index=False, header=True)
br.fit(X_train,y_train)
br_prediction=np.exp(br.predict(X_test))
br_submission = pd.concat([test_df['Id'], pd.Series(br_prediction, name='SalePrice')], axis=1)
br_submission.to_csv('br_try1.csv', index=False, header=True)
omp.fit(X_train,y_train)
omp_prediction=np.exp(omp.predict(X_test))
omp_submission = pd.concat([test_df['Id'], pd.Series(omp_prediction, name='SalePrice')], axis=1)
omp_submission.to_csv('omp_try1.csv', index=False, header=True)
ridge.fit(X_train,y_train)
ridge_prediction=np.exp(ridge.predict(X_test))
ridge_submission = pd.concat([test_df['Id'], pd.Series(ridge_prediction, name='SalePrice')], axis=1)
ridge_submission.to_csv('ridge_try1.csv', index=False, header=True)
gbr.fit(X_train,y_train)
gbr_prediction=np.exp(gbr.predict(X_test))
gbr_submission = pd.concat([test_df['Id'], pd.Series(gbr_prediction, name='SalePrice')], axis=1)
gbr_submission.to_csv('gbr_try1.csv', index=False, header=True)
models = {
    "br": br,
    "catboost": catboost,
    "omp": omp,
    'GBR':gbr,
    "ridge": ridge

}
predictions = (
    0.4 * np.exp(models['br'].predict(X_test)) +
    0.4 * np.exp(models['catboost'].predict(X_test)) +
    0.1 * np.exp(models['omp'].predict(X_test)) +
     0.05 * np.exp(models['GBR'].predict(X_test)) +
    0.05 * np.exp(models['ridge'].predict(X_test))

)
ensemble_submission = pd.concat([test_df['Id'], pd.Series(predictions, name='SalePrice')], axis=1)
ensemble_submission.to_csv('ensemble_try_2.csv', index=False, header=True)
