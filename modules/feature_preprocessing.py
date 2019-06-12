# Required libraries
import pandas as pd
import logging as log
import time
import numpy as np
from scipy import signal
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


"""
Function to apply sliding windows on a proteins dataset. It uses Gaussian Filtering
"""
def sliding_window(data, k, sd):
    """
    REQUIRE:
    import pandas as pd
    import numpy as np
    import signal from scipy

    INPUT:
    data = dataframe of main features
    k = the size of a window (int)
    sd = the standard deviation of the gaussian filter (float)

    OUTPUT:
    A dataframe with sliding windows applied
    """

    # Define starting time of the function
    start = time.time()
    #Set variables
    df_windows = data.copy()

    #Cycle for every protein
    for pdb_id in data.PDB_ID.unique():
        #Cycle for every chain in a given protein
        for chain in set(data.CHAIN_ID[data.PDB_ID == pdb_id].unique()):

            #Work on a reduced dataset: we apply sliding windows for every chain
            df_sliced = df_windows[(data.PDB_ID == pdb_id)
                                   & (data.CHAIN_ID == chain)]

            # SET PDB_ID, CHIAN_ID and RES_ID to a separated df, we are not going to apply gaussian filter on them
            info_sliced = df_sliced.iloc[:, 0:3]

            #Shortcut name for lengths
            chain_len = len(data.CHAIN_ID[(data.PDB_ID == pdb_id)
                                        & (data.CHAIN_ID == chain)])

            #Apply a symmatric mirroring at the start of the chain of size k//2
            df_windows_start = pd.DataFrame(np.array(df_sliced.iloc[1:(k//2+1), ]),
                                            index=np.arange(-k//2 + 1, 0, step = 1),
                                            columns=list(data.columns)).sort_index()

            #Apply a symmatric mirroring at the end of the chain of k//2

            df_windows_end = pd.DataFrame(np.array(df_sliced.iloc[chain_len-(k//2 + 1):chain_len-1, ]),
                                          index=np.arange(chain_len-1 + k//2,chain_len-1, step = -1),
                                          columns=list(data.columns)).sort_index()

            #Now we merge reunite this dataframe
            df_with_start_sym = df_windows_start.append(df_sliced)
            df_win_k = df_with_start_sym.append(df_windows_end)

            ### MAIN: COMPUTE GAUSSIAN FILTER OF GIVEN DATAFRAME
            sliced = df_win_k.iloc[:, 3:]
            window = signal.gaussian(k, std = sd)
            sliced = sliced.rolling(window = k, center = True).apply(lambda x: np.dot(x,window)/k)

            # Reunite filtered features with PDB_ID, CHAIN_ID, RES_ID
            tot_sliced = pd.merge(info_sliced, sliced.iloc[0:chain_len+k//2,:],
                                  right_index=True, left_index=True) #here is chain_len + k//2

            ### Update the dataframe with the filtered features of given chain
            df_windows[(df_windows.PDB_ID == pdb_id) & (df_windows.CHAIN_ID == chain)] = tot_sliced

    # Debug time
    log.debug('Window sliding took {}'.format(time.time() - start))
    # Return "window slided" dataframe
    return df_windows
