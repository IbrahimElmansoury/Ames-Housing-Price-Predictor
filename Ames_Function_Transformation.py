from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Input:  Year Built, Year Remod/Add, Gr Liv Area,
            Total Bsmt SF, Wood Deck SF, Open Porch SF, TotRms AbvGrd
    Output: House_Age, Remodel_Age, Is_Remodeled,
            Total_Home_Area, Total_Porch_Area, Avg_Room_Size
    """
    Base_Year = 2010

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        # temporal
        X['House_Age'] = self.Base_Year - X['Year Built']
        X['Remodel_Age'] = self.Base_Year - X['Year Remod/Add']
        X['Is_Remodeled'] = (X['Year Built'] != X['Year Remod/Add']).astype(int)

        # aggregates
        X['Total_Home_Area'] = X['Gr Liv Area'] + X['Total Bsmt SF']
        X['Total_Porch_Area'] = X['Wood Deck SF'] + X['Open Porch SF']
        X['Avg_Room_Size'] = X['Gr Liv Area'] / X['TotRms AbvGrd'].replace(0, 1)

        out_cols = [
            'House_Age', 'Remodel_Age', 'Is_Remodeled',
            'Total_Home_Area', 'Total_Porch_Area', 'Avg_Room_Size',
        ]
        return X[out_cols].values

    def get_feature_names_out(self, input_features=None):
        return [
            'House_Age', 'Remodel_Age', 'Is_Remodeled',
            'Total_Home_Area', 'Total_Porch_Area', 'Avg_Room_Size',
        ]