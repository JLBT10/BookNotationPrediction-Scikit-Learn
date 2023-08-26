from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import unidecode
import numpy as np
import warnings


def month_year(df,columns):
    
    df["year"]=df[columns].apply(lambda x : int(x.split("/")[-1]))
    df["month"]=df[columns].apply(lambda x : int(x.split("/")[0])) 
    return df

def drop_outliers(df):
    mask=(df.ratings_count==0) | (df.average_rating==0)
    return df[~mask]

class BooksProcessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X=pd.read_csv("books.csv",on_bad_lines='skip')
        X_transformed = X.copy()
        
        # Drop duplicate titles
        X_transformed = X_transformed.drop(X_transformed[X_transformed['title'].duplicated(keep='first')].index)
        
        # Process num_pages
        X_transformed['num_pages'] = X_transformed['num_pages'].replace(0, np.mean(X_transformed['num_pages']))
        high_pages_filtered = X_transformed[X_transformed['num_pages'] > 1000]
        X_transformed = X_transformed.drop(high_pages_filtered.index, axis=0)
        
        # Process publication_date
        X_transformed = month_year(X_transformed, 'publication_date')
        
        # Process Authors
        X_transformed = X_transformed[X_transformed['authors'] != 'NOT A BOOK']
        
        # Process language
        X_transformed['language'] = X_transformed['language'].apply(lambda x: 'eng' if 'eng' in x or 'en' in x else x)
        
        # Process rating_counts
        X_transformed = drop_outliers(X_transformed)
        
        # Drop unused columns
        #X_transformed.drop(['publication_date', 'isbn13', 'title','isbn', 'bookID','authors','average_rating','language','publisher'], inplace=True, axis=1)
        
        return X_transformed



def compute_average_notation(authors_list,lookup_authors,mean_notation):
    count=0
    sum_notation=0
    for name in authors_list:
        if name in lookup_authors:
            count+=1
            sum_notation+=lookup_authors[name]
        else :
            count+=1
            sum_notation+=mean_notation

    return np.round(sum_notation/count,2)

class AuthorsNotationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        from unidecode import unidecode
        X_transformed = X.copy()
        # Load the authors info and compute the lookup and mean notation
        info_authors = pd.read_json('goodreads_book_authors.json', lines=True)
        self.lookup_authors = dict(zip(info_authors['name'], info_authors['average_rating']))
        self.mean_notation = np.round(np.mean(info_authors['average_rating']), 2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pd.set_option('mode.chained_assignment', None)

            X_transformed['authors'] = X_transformed['authors'].apply(lambda x: unidecode(x))
            X_transformed['authors_split'] = X_transformed['authors'].apply(lambda x: x.split('/'))
            X_transformed['authors_average_notation'] = X_transformed['authors_split'].apply(lambda x: compute_average_notation(x, self.lookup_authors, self.mean_notation))
            #X_transformed.drop(['authors_split','authors'], inplace=True, axis=1)
        
        return X_transformed


class AuthorsFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        X_transformed['rate_ratings'] = 1 - (X_transformed['text_reviews_count']) / X_transformed['ratings_count']
        X_transformed['rate_ratings_authors'] = X_transformed['rate_ratings'] * X_transformed['authors_average_notation']
        
        # You can choose to drop columns here if needed
        #X_transformed.drop(['rate_ratings'], axis=1, inplace=True)
        
        return X_transformed


class ColumnRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.drop(self.columns_to_remove, axis=1)
        return X_transformed


class StandardScalerWithNames(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.columns = X.columns
        return self
    
    def transform(self, X):
        scaled_data = self.scaler.transform(X)
        scaled_df = pd.DataFrame(scaled_data, columns=self.columns)
        return scaled_df



        