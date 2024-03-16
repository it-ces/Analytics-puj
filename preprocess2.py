# Preprocessing Module

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def reduce_categories(X, min_percent):
    # if the relative share is lesser than min_percent put the label 'Another'
    X_transformed = X.copy()
    for var in X_transformed.columns:
        cats = X_transformed[var].value_counts(normalize=True).to_dict()
        to_replace = [key for key in cats if cats[key]<min_percent]  
         # Convert categories that not appear at least minpercent to <Another>
        X_transformed[var] = X_transformed[var].replace(to_replace=to_replace, value='Another')
    return X_transformed.to_numpy()
      
class reduceCategories(BaseEstimator, TransformerMixin):
    def __init__(self, min_percent=0.05):
        self.min_percent = min_percent

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        return reduce_categories(pd.DataFrame(X_transformed), self.min_percent)

 

# To use reduce_categories in a pipeline!!




# Drop columns with NaN's
def drop_colNaN(df_, min_percent):
    # If the column have lesser of threshold complete data 
    # will be drop of the dataset...
    # threshold = 1,indicates that keep only variables without nans...
    df = df_.copy()
    N  = df.shape[0] 
    keep = [var for var in df.columns if df[var].notnull().sum()/N >= min_percent]
    return df[keep].to_numpy()


# Drop columns to uses in a PipeLine!!!
class drop_ColumnsNan(BaseEstimator, TransformerMixin):
    def __init__(self, min_percent=0.8):
        self.min_percent = min_percent

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        return drop_colNaN(pd.DataFrame(X_transformed), self.min_percent)



############################ Preprocess...
    
# Algortihm Table One

from scipy import stats

def classify_vars(df):
    categorial, nonormal, normal = [],[],[]
    for t in df.columns:
        if df[t].dtypes=="object" or df[t].dtypes.name=='category':
            categorial.append(t)
        if df[t].dtypes=="int64" or df[t].dtypes=="float64":
            n,p = stats.shapiro(df[t])
            if p<0.05:
                nonormal.append(t)
            else: 
                normal.append(t)
    return categorial, nonormal, normal