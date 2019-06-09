# Required libraries
import pandas as pd
import logging as log
from sklearn.feature_extraction.text import CountVectorizer

# Given an iterable (list or Series), turns it into a bag of words matrix (DataFrame)
def get_bow(iterable, vocabulary=None, prefix=''):
    # Turn vocabulary words lowercase, as required by CountVectorizer
    if vocabulary:
        vocabulary = [v.lower() for v in vocabulary]
    # Apply CountVectorizer with given vocabulary
    cv = CountVectorizer(vocabulary=vocabulary)
    # Compute BOW matrix
    bow = pd.DataFrame(cv.fit_transform(iterable).toarray(), columns=['{}{}'.format(prefix, f.upper()) for f in cv.get_feature_names()])
    # Return computed bag of words
    return bow

def get_bow_residues(residues, vocabulary=None, prefix='RES_NAME_'):
    return get_bow(structs, vocabulary, prefix)

def get_bow_structures(structs, vocabulary=None, prefix='STRUCT_'):
    return get_bow(structs, vocabulary, prefix)

def get_bow_edge_loc(structs, vocabulary=None, prefix='EDGE_LOC_'):
    return get_bow(structs, vocabulary, prefix)

def get_bow_edge_type(structs, vocabulary=None, prefix='EDGE_TYPE'):
    return get_bow(structs, vocabulary, prefix)

# Given a DataFrame and a column, removes the column and adds BOW columns computed from the latter
def replace_bow(df, col, vocabulary=None, prefix='', drop=False):
    # Retrieve column which will be removed
    removed = df[col]
    # Delete column from dataframe if requested
    if drop:
        df = df.drop(col, axis=1, inplace=False)
    # Compute BOW
    bow = get_bow(removed, vocabulary=vocabulary, prefix=prefix)
    # Concatenate DataFrames
    df = pd.concat([df, bow], axis=1)
    # Return computed DataFrame
    return df
