import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer, ArbitraryOutlierCapper
from sklearn.preprocessing import PowerTransformer, StandardScaler, Binarizer
from Ames_Function_Transformation import FeatureEngineer
from feature_engine.encoding import RareLabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib





df = pd.read_csv('AmesHousing_Top25.csv')

#remove 3 outliers
outliers_index = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 300000)].index
df.drop(outliers_index, axis=0, inplace=True)

#remow rows cols_with_missing_1_values
cols_with_missing_1_values = df.columns[df.isnull().sum() == 1]
df.dropna(subset=cols_with_missing_1_values, inplace=True)

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['Overall Qual'])


Lot_Frontage_Col = ['Lot Frontage']
Lot_Frontage_Pipe = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=10, random_state=42, add_indicator=True)),
    ('winsorizer', Winsorizer(capping_method='gaussian',fold=3, tail='both')),
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True))
])


Mas_Vnr_Area_col = ['Mas Vnr Area']
Mas_Vnr_Area_Pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True))
])

Skewed_Gaussian_cols = ['Lot Area', 'Gr Liv Area', '1st Flr SF']
Skewed_Gaussian_Pipe = Pipeline(steps=[
    ('winsorizer', Winsorizer(capping_method='gaussian', fold=3, tail='both')),
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True))
])

MultiModal_cols = ['Wood Deck SF', 'Open Porch SF', 'BsmtFin SF 1', 'Garage Area']
MultiModal_Pipe = Pipeline(steps=[
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True))
])

Binary_Flag_cols = ['Mas Vnr Area','Wood Deck SF', 'Open Porch SF', 'BsmtFin SF 1']
Binary_Flag_Pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('binarizer', Binarizer(threshold=0.0))
])


feature_eng_cols = ['Year Built', 'Year Remod/Add','Gr Liv Area', 'Total Bsmt SF','Wood Deck SF', 'Open Porch SF','TotRms AbvGrd',]
feature_eng_pipe = Pipeline(steps=[
    ('FeatureEngineer', FeatureEngineer()),
    ('scaler',StandardScaler()),
])


Discrete_cols = ['Garage Cars', 'TotRms AbvGrd', 'Overall Qual', 'Full Bath', 'Fireplaces']
Discrete_Pipe = Pipeline(steps=[
    ('capper', ArbitraryOutlierCapper( max_capping_dict={'Garage Cars': 3,'TotRms AbvGrd': 10}, min_capping_dict=None)),
    ('scaler', StandardScaler())
])

Rare_Label_cols = ['Neighborhood', 'MS Zoning', 'House Style']
Rare_Label_Pipe = Pipeline(steps=[
    ('encoder', RareLabelEncoder(tol=0.05, n_categories=1, replace_with='Rare')),
    ('OHE', OneHotEncoder(drop= 'first',handle_unknown='ignore', sparse_output= False))
])

nominal_cols = ['Bldg Type', 'Central Air']
nominal_pipe = Pipeline(steps=[
    ('OHE', OneHotEncoder(drop= 'first',handle_unknown='ignore', sparse_output= False)),
])

Exter_ord = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
Kitchen_ord = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
Bsmt_ord = ['No_Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

ordinal_cols = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual']
ordinal_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='No_Basement')),
    ('Ordinal', OrdinalEncoder(categories=[Exter_ord,Kitchen_ord,Bsmt_ord] ,handle_unknown='use_encoded_value', unknown_value=-1)),
])

PreProcessing = ColumnTransformer(transformers=[
    ('Lot_Frontage_Pipe', Lot_Frontage_Pipe, Lot_Frontage_Col),
    ('Mas_Vnr_Area_Pipe', Mas_Vnr_Area_Pipe, Mas_Vnr_Area_col),
    ('Skewed_Gaussian_Pipe', Skewed_Gaussian_Pipe, Skewed_Gaussian_cols),
    ('MultiModal_Pipe', MultiModal_Pipe, MultiModal_cols),
    ('Binary_Flag_Pipe', Binary_Flag_Pipe, Binary_Flag_cols),
    ('feature_eng_pipe', feature_eng_pipe, feature_eng_cols),
    ('Discrete_Pipe', Discrete_Pipe, Discrete_cols),
    ('Rare_Label_Pipe', Rare_Label_Pipe, Rare_Label_cols),
    ('nominal_pipe', nominal_pipe, nominal_cols),
    ('ordinal_pipe', ordinal_pipe, ordinal_cols),
    ], remainder='drop'
)

log_target_model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)

master_pipeline = Pipeline(steps=[
    ('preprocessor', PreProcessing),
    ('model', log_target_model)
])


master_pipeline.fit(X_train, y_train)

y_pred = master_pipeline.predict(X_test)


cross_val = cross_val_score(master_pipeline, X_train, y_train,scoring="r2" ,cv=5)
print(f"Cross Validation Score: {cross_val.mean():.4f} +/- {cross_val.std():.4f}")


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

joblib.dump(master_pipeline, 'ames_housing_model.pkl')
print("ames_housing_model.pkl saved successfully")
