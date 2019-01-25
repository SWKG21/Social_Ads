import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../")
from problem import get_train_data, get_test_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse



class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        return X_df
