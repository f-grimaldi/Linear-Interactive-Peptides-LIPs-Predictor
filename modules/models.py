# Import required modules
import logging
# Import default libraries
import pandas as pd
import numpy as np
import requests
import json
import zipfile
import time
import warnings
# Import BioPython utils
from Bio.PDB import PDBList, calc_angle, calc_dihedral, PPBuilder, is_aa, PDBIO, NeighborSearch, DSSP, HSExposureCB
from Bio.PDB.PDBParser import PDBParser
# Import other methods
from sklearn.feature_extraction.text import CountVectorizer
from scipy import signal
#Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#Metrics
from sklearn import metrics


"""
Function to apply sliding windows on a proteins dataset. It uses Gaussian Filtering
"""
def sliding_windows(data, window, std, ignore_warnings = False, get_time = False):
    """
    REQUIRE:
    import pandas as pd
    import numpy as np
    import signal from scipy

    INPUT:
    data = dataframe of main features
    window = the size of a window (int)
    std = the standard deviation of the gaussian filter (float)
    get_time = If true print the time needed to complete the procedure (default False)
    ignore_warnings = If true it will disable the warnings (default False)

    OUTPUT:
    A dataframe with sliding windows applied
    """

    #Ignore warnings
    if ignore_warnings:
        warnings.filterwarnings("ignore")

    start = time.time()
    df_windows = data.copy()
    k = window
    sd = std

    #Cycle for every protein
    for pdb_id in data.PDB_ID.unique():
        #Cycle for every chain in a given protein
        for chain in set(data.CHAIN_ID[data.PDB_ID == pdb_id].unique()):

            #Work on a reduced dataset
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

            #Append symmatric mirroring into one dataframe
            df_with_start_sym = df_windows_start.append(df_sliced)
            df_win_k = df_with_start_sym.append(df_windows_end)

            ### MAIN: COMPUTE GAUSSIAN FILTER OF GIVEN DATAFRAME
            sliced = df_win_k.iloc[:, 3:]
            window = signal.gaussian(k, std = sd) #Here put k
            sliced = sliced.rolling(window = k, center = True).apply(lambda x: np.dot(x,window)/k) #here put k
            #sliced = sliced.rolling(window = 3, center = True, win_type = 'gaussian').sum(std=1)

            # Reunite filtered features with PDB_ID, CHAIN_ID, RES_ID
            tot_sliced = pd.merge(info_sliced, sliced.iloc[0:chain_len+k//2,:],
                                  right_index=True, left_index=True) #here is chain_len + k//2

            ### Update the dataframe with the filtered features of given chain
            df_windows[(df_windows.PDB_ID == pdb_id) & (df_windows.CHAIN_ID == chain)] = tot_sliced

    #If get_time print time needed to end process
    if get_time:
        print(time.time() - start)

    #Restore warnings
    warnings.filterwarnings('default')

    return df_windows



"""
Function that take a dataset and a classfier, possibly after sliding windows and return some score and information
of the classifier with a Leave One Out Cross Validation Protein Based.
For every protein it use that protein as test set and the others fot training
"""
def loo_cv(data, target, clf,
           bad_condition = (0.75, 0.75),
           ign_warnings = False, get_times = True,
           verbose = False):

    """
    REQUIRE:
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    'module of classifier'

    INPUT:
    data = dataframe of features
    target = the array of labels (LIP)
    clf = the build in classifier object
    bad_condition = (tuple) the values of balanced accuracy and f1 score that the model need to pass in order to not be considered bad.
    ign_warnings = If true it disablethe warnings (default False)
    get_time = If true print the time needed to complete the procedure (default True)
    verbose = If True, it will print the result for every protein wih confusion matrix (default False)


    OUTPUT:
    A list containing:
    1. List of balanced Accuracies of every protein.
    2. Mean of (1.)
    3. List of f1 score of every proteinself.
    4. Mean of (3.)
    5. List of tuples of proteins having bad results (see bad_condition). Every tuple has:
        a. PDB_ID of the protein
        b. Balance accuracy
        c. f1 score
    """

    #STEP I: set time and warnings
    if ign_warnings:
        warnings.filterwarnings("ignore")
    start = time.time()

    #STEP II: Create input and target (X,y)
    df_clf = data.copy()
    df_clf['LIP'] = target

    #STEP III: Create varaibles to save results
    BAL_ACC = [] #List of balanced accuracy for every model tested
    F1_SCORE = []
    BAD_PERF = [] #List of tuples (pdb_id, balanced accuracy, f1 score) of proteins that have 'bad condtion' parameters
    #Keep trak of iteration (for verbose)
    i = 1

    #STEP IV:
    #For every protein use that protein as test set and others as training
    for pdb_id in df_clf.PDB_ID.unique():

        #Set train and test dataframe
        df_train = df_clf.copy().iloc[:, 3:]
        df_target = df_clf[df_clf.PDB_ID == pdb_id].iloc[:, 3:]
        df_train.drop(list(df_target.index), inplace = True)

        #Create X_train, y_train
        y_train = np.array(df_train.loc[:, 'LIP'])
        df_train.drop(['LIP'], axis = 1, inplace = True)
        X_train = np.array(df_train)

        #Create X_test, y_test
        y_test = np.array(df_target.loc[:, 'LIP'])
        df_target.drop(['LIP'], axis = 1, inplace = True)
        X_test = np.array(df_target)

        #FIT MODEL
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)

        #Compute metrics
        acc_score = np.round(metrics.accuracy_score(y_test, y_pred), 3)
        bal_acc_score = np.round(metrics.balanced_accuracy_score(y_test, y_pred), 3)
        f1_score = np.round(metrics.f1_score(y_test, y_pred), 3)
        precision_score = np.round(metrics.precision_score(y_test, y_pred), 3)
        recall_score = np.round(metrics.recall_score(y_test, y_pred), 3)

        #Append
        BAL_ACC.append(bal_acc_score)
        F1_SCORE.append(f1_score)
        if bal_acc_score < 0.75 or f1_score < 0.75:
            BAD_PERF.append((pdb_id, bal_acc_score, f1_score))

        #If verbose print a series of information
        if verbose:
            print('Ieration number:', i)
            print(pdb_id)
            print("Accuracy:", acc_score)
            print("Balanced Accuracy:", bal_acc_score)
            print("Precision:", precision_score)
            print("Recall:", recall_score)
            print("F1 score:", f1_score)

            print(metrics.confusion_matrix(y_test, y_pred))
            print('______________________________________________________')
        i += 1


    #STEP VI: restore warnings, print time, return
    if ign_warnings:
        warnings.filterwarnings('default')
    if get_times:
        print('Time taken: {}\n'.format(time.time() - start))

    return [BAL_ACC, np.mean(BAL_ACC), F1_SCORE, np.mean(F1_SCORE), BAD_PERF]
