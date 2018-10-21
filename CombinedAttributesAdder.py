from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, poly_features=None, combined_features = None):         
        self.poly_features = poly_features
        self.combined_features = combined_features
        self.name_combined_features = []
        self.name_created_poly_features = []
        self.created_poly_features = None
    def fit(self, X, Y=None):
        return self
    def transform(self, X, Y=None):
        if np.any(self.poly_features):
            poly_transformer = PolynomialFeatures(degree = 3)
            for _, value in self.poly_features.items():
                poly_features = X[:, value[1]]
                poly_transformer.fit(poly_features)
                self.created_poly_features = poly_transformer.transform(poly_features) 
                self.name_created_poly_features = poly_transformer.get_feature_names(input_features = value[0])
            X = np.c_[X, self.created_poly_features]
        if np.any(self.combined_features):
            for key, value in self.combined_features.items():
                self.name_combined_features.append(key)
                new_feature = X[:, value[0]] / X[:, value[1]]
                X = np.c_[X, new_feature]
        return X
           
