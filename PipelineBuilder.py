from DataFrameSelector import *
from future_encoders import *
from CombinedAttributesAdder import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

class PipelineBuilder():

    def buil_cat_pipeline(self, cat_attribs):
        return Pipeline([
                ('selector', DataFrameSelector(cat_attribs)),
                ('cat_encoder', OneHotEncoder(sparse=False)),
            ])
    
    def buil_numeric_pipeline(self, num_attribs):
        return  Pipeline([
                    ('selector', DataFrameSelector(num_attribs)),
                    ('imputer', Imputer(strategy = "median")),
                    ('attribs_adder', CombinedAttributesAdder()),
                    ('std_scaler', StandardScaler()),
                ])

    def build_full_pipeline(self, cat_attribs, num_attribs):
        return FeatureUnion(transformer_list=[
                    ("num_pipeline", self.buil_numeric_pipeline(num_attribs)),
                    ("cat_pipeline", self.buil_cat_pipeline(cat_attribs)),
                ])

    def build_full_pipeline_with_predictor(self, clf, cat_attribs, num_attribs):
        return Pipeline([
                ("preparation", self.build_full_pipeline(cat_attribs, num_attribs)),
                ("clf", clf)
            ])

