from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class LotFrontageImputer(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        # Вычисление медианы для каждого района и сохранение их в атрибуте объекта
        self.lotfrontage_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        # Заполнение пропущенных значений медианами, соответствующими районам
        X_copy['LotFrontage'] = X_copy.apply(lambda row: self.lotfrontage_medians_[row['Neighborhood']] if pd.isnull(row['LotFrontage']) else row['LotFrontage'], axis=1)
        return X_copy
    

class ColumnDroper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)