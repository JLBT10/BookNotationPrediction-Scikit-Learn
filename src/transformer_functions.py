
import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json

def month_year(df,columns):
    ''' Takes DataFrame and a date columns (9/30/2003 and returns the DataFrame with two added separate columns(year and month) '''
    df["year"]=df[columns].apply(lambda x : int(x.split("/")[-1]))
    df["month"]=df[columns].apply(lambda x : int(x.split("/")[0])) 
    return df

def drop_outliers(df):
    '''Drop all the rows where ratings_counts=0 and average_rating==0'''
    mask=(df.ratings_count==0) & (df.average_rating==0)
    return df[~mask]

class BooksProcessingTransformer(BaseEstimator, TransformerMixin):
    '''Process the DataFrame books'''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        X_transformed = X.copy()
        
        # Drop duplicate titles
        #X_transformed = X_transformed.drop(X_transformed[X_transformed['title'].duplicated(keep='first')].index)
        
        # Process num_pages
        X_transformed['num_pages'] = X_transformed['num_pages'].replace(0, np.mean(X_transformed['num_pages']))
        high_pages_filtered = X_transformed[X_transformed['num_pages'] > 1000]
        X_transformed = X_transformed.drop(high_pages_filtered.index, axis=0)
        
        # Process publication_date
        X_transformed = month_year(X_transformed, 'publication_date')
        
        # Process Authors
        X_transformed = X_transformed[X_transformed['authors'] != 'NOT A BOOK']
        
        # Process language
        X_transformed['language_code'] = X_transformed['language_code'].apply(lambda x: 'eng' if 'eng' in x or 'en' in x else x)
        
        # Process rating_counts
        X_transformed = drop_outliers(X_transformed)
        
        # Drop unused columns
        #X_transformed.drop(['publication_date', 'isbn13', 'title','isbn', 'bookID','authors','average_rating','language','publisher'], inplace=True, axis=1)
        
        return X_transformed



def compute_average_notation(authors_list,lookup_authors,mean_notation):

    ''' compute the average notation for group of authors'''

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

    'Add the notation of authors as features in the DF'

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        from unidecode import unidecode
        X_transformed = X.copy()

        with open('./../datasets/dict_goodreads_book_authors.json', "r") as json_file:
            self.lookup_authors = json.load(json_file)

        self.mean_notation = sum(self.lookup_authors.values()) / len(self.lookup_authors)


        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pd.set_option('mode.chained_assignment', None)

            X_transformed['authors'] = X_transformed['authors'].apply(lambda x: unidecode(x))
            X_transformed['authors_split'] = X_transformed['authors'].apply(lambda x: x.split('/'))
            X_transformed['authors_average_notation'] = X_transformed['authors_split'].apply(lambda x: compute_average_notation(x, self.lookup_authors, self.mean_notation))
        
        return X_transformed


class AuthorsFeaturesTransformer(BaseEstimator, TransformerMixin):
    '''Engineering a new feature  : 
        when transform -->
        X_transformed['rate_ratings'] = (X_transformed['ratings_count'] - X_transformed['text_reviews_count']) / X_transformed['ratings_count']
        --> Ratio of rating count free of text reviews over total ratings count
        and then we multiply that ratio by the average notation of auhtors and we get the part of the notation that we can attribuate to the ratio
     '''

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed_mask= X_transformed['ratings_count']!=0
        X_transformed.loc[X_transformed_mask, 'ratio_of_text_reviews'] = X_transformed['text_reviews_count'] / X_transformed['ratings_count']
        X_transformed.loc[~X_transformed_mask,'ratio_of_text_reviews']=0
        #X_transformed['rate_ratings_authors'] = X_transformed['rate_ratings'] * X_transformed['authors_average_notation']
        
        # You can choose to drop columns here if needed
        #X_transformed.drop(['rate_ratings'], axis=1, inplace=True)
        
        return X_transformed


class ColumnRemoverTransformer(BaseEstimator, TransformerMixin):
    '''removes specified columns in list'''
    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.drop(self.columns_to_remove, axis=1)
        return X_transformed


class StandardScalerWithNames(BaseEstimator, TransformerMixin):
    '''Modified the StandardScaler() so it can returns a pd.DataFrame and not a numpy array '''
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


def get_sum_books(df):
    authors = df.copy()
    authors['authors'] = authors['authors'].astype(str)
    authors['authors'] = authors['authors'].apply(lambda x: x)
    authors['authors_split'] = authors['authors'].apply(lambda x: x.split('/'))
    authors = authors.explode('authors_split')

    most_books_written = authors.groupby('authors_split')['title'].count().sort_values(ascending=False)
    most_books_written = most_books_written.rename('nb_written_book').reset_index()
    
    return most_books_written,authors


def plot_bar(classement,nb,head_or_tail):
  if head_or_tail=='head':
    classement=classement.sort_values(by='nb_written_book',ascending=False)
    # Tracer le diagramme à barres
    sns.barplot(x=classement.authors_split[0:nb], y=classement.nb_written_book[0:nb])

    # Rotation des étiquettes
    plt.xticks(rotation='vertical')

    # Ajouter des titres et des étiquettes d'axe
    plt.xlabel('Authors')
    plt.ylabel('number of written books')
    plt.title('Top 10 of authors with the most wwritten book')

    # Afficher le diagramme
    plt.show()
  else :
    classement=classement.sort_values(by='nb_written_book',ascending=True)
    sns.barplot(x=classement.authors_split[0:nb],y= classement.nb_written_book[0:nb])

    # Rotation des étiquettes
    plt.xticks(rotation='vertical')

    # Ajouter des titres et des étiquettes d'axe
    plt.xlabel('Authors')
    plt.ylabel('number of written books')
    plt.title('Top 10 of authors with the most wwritten book')

    # Afficher le diagramme
    plt.show()

        
