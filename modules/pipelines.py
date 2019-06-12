# Import libraries
import pandas as pd
import numpy as np
import os
import logging
import re
from joblib import dump, load
# Import custom libraries
from modules.feature_extraction import *
from modules.feature_preprocessing import *
# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Define list of available classifiers
available_clf = ['RandomForestClassifier', 'LogisticRegression', 'MLPCLassifier', 'KNeighborsClassifier', 'SVC', 'QuadraticDiscriminantAnalysis']

# Define main pipeline used by the program
def main_pipeline(pdb_ids=[], config={}):
    # Retrieve PDB file
    download_PDB(pdb_ids,  pdb_dir=config.get('pdb_dir'))
    # Retrieve PDB data from PDB files
    ds_residues = get_PDB(pdb_ids, valid_chains=config.get('valid_chains', None), pdb_dir=config.get('pdb_dir'))
    # Compute Bag Of Words of RES_NAME feature
    ds_residues = replace_bow(ds_residues,
                                col='RES_NAME',
                                vocabulary=config.get('vocabularies').get('res_name'),
                                prefix='RES_NAME_')
    # Debug
    logging.debug(ds_residues.head())

    # Retrieve DSSP features
    ds_dssp = get_DSSP(pdb_ids, pdb_dir=config.get('pdb_dir'), dssp_path=config.get('dssp_path'))
    # Map dssp secondary structure
    if config.get('map_struct'):
        ds_dssp['SEC_STRUCT'] = ds_dssp['SEC_STRUCT'].map(config.get('map_struct'))
    # Add secondary structure encoding to dssp dataset
    ds_dssp = replace_bow(ds_dssp,
                            col='SEC_STRUCT',
                            vocabulary=config.get('vocabularies').get('struct_name'),
                            prefix='SEC_STRUCT_')
    # Merge DSSP features into residues dataset
    ds_residues = ds_residues.merge(ds_dssp, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
    # Handle null values: drop
    ds_residues = ds_residues.dropna(how='any', axis=0)
    # Debug
    logging.debug(ds_residues.head())

    # Define empty list of files which will be donwloaded, identified by a pdb id
    ring_downloads = []
    # If forced download flag is enabled, set all ring files to be downloaded
    if config.get('ring_force'):
        ring_downloads = pdb_ids
    else:
        # Define all files
        ring_files = ['{}/{}_network.zip'.format(config.get('ring_dir'), pdb_id) for pdb_id in pdb_ids]
        # Append only PDB ids of files which have not already been downloaded
        ring_downloads = [pdb_id for pdb_id in pdb_ids if not os.path.isfile('{}/{}_network.zip'.format(config.get('ring_dir'), pdb_id))]
    # Download RING files
    download_RING(ring_downloads, ring_dir=config.get('ring_dir'))

    # Retrieve RING features
    ds_RING = get_RING(pdb_ids, pdb_dir=config.get('pdb_dir'), ring_dir=config.get('ring_dir'), contact_threshold=config.get('contact_threshold'))
    # Compute edge features from RING dataset
    ds_RING_edges = get_RING_edges(ds_RING)
    # Merge edge features into residues dataset
    ds_residues = ds_residues.merge(ds_RING_edges, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
    # Handle null values
    ds_residues.fillna({'EDGE_LOC': '', 'EDGE_TYPE': ''}, inplace=True)
    # Debug
    logging.debug(ds_residues.head())

    # Compute Bag Of Words of EDGE_LOC feature
    ds_residues = replace_bow(ds_residues,
                                col='EDGE_LOC',
                                vocabulary=config.get('vocabularies', {}).get('edge_loc'),
                                prefix='EDGE_LOC_')
    # Compute Bag Of Words of EDGE_TYPE feature
    ds_residues = replace_bow(ds_residues,
                                col='EDGE_TYPE',
                                vocabulary=config.get('vocabularies', {}).get('edge_type'),
                                prefix='EDGE_TYPE_')
    # Debug
    logging.debug(ds_residues.head())

    # Compute contact features from RING dataset
    ds_RING_contacts = get_RING_contacts(ds_RING)
    # Merge contact features into residues dataset
    ds_residues = ds_residues.merge(ds_RING_contacts, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
    # Handle null values
    ds_residues.fillna({'INTRA_CONTACTS': 0.0, 'INTER_CONTACTS': 0.0, 'INTRA_INTER_CONTACTS': 0.0}, inplace=True)
    # Debug
    logging.debug(ds_residues.head())

    # Define a dataset before applying sliding window
    # Define the set of all columns
    col = set(ds_residues.columns)
    # Remove useless columns
    col -= set(['MODEL_ID', 'RES_NAME', 'SEC_STRUCT', 'EDGE_TYPE', 'EDGE_LOC', 'LIP', 'LIP_SCORE'])
    # col -= set(['PDB_ID', 'MODEL_ID', 'CHAIN_ID', 'RES_ID', 'RES_NAME', 'SEC_STRUCT', 'EDGE_TYPE', 'EDGE_LOC', 'LIP', 'LIP_SCORE'])
    # Delete INTRA_INTER_CONTACTS of the two columns INTRA_CONTACTS and INTER_CONTACTS
    if not config.get('intra_inter_ratio'):
        col -= set(['INTRA_INTER_CONTACTS'])
    else:
        col -= set(['INTRA_CONTACTS', 'INTER_CONTACTS'])
    # Delete EDGE_LOC features
    if config.get('no_edge_loc') or config.get('no_edge'):
        col -= set([c for c in col if re.match('^EDGE_LOC_', c)])
    # Delete EDGE_TYPE features
    if config.get('no_edge_type') or config.get('no_edge'):
        col -= set([c for c in col if re.match('^EDGE_TYPE_', c)])
    # Update dataset for prediction (keeps columns in the correct order)
    ds_predict = ds_residues.loc[:, [c for c in list(ds_residues.columns) if c in col]]
    # Execute sliding window averaging on main dataset
    ds_predict = sliding_window(ds_predict, k=config.get('window_size'), sd=config.get('window_std'))

    # Define dataframe for prediction
    col -= set(['PDB_ID', 'CHAIN_ID', 'RES_ID'])
    ds_predict = ds_predict.loc[:, [c for c in list(ds_residues.columns) if c in col]]

    # Debug
    logging.debug(ds_predict.head())
    logging.debug(ds_predict.columns)

    # Return complete and prediction-suited dataframe
    return ds_residues, ds_predict


# Define pipeline used for predictions
def predict_pipeline(pdb_ids, config={}):
    # Retrieve datasets for prediction
    ds_residues, ds_predict = main_pipeline(pdb_ids, config)
    # Debug
    logging.debug(ds_predict.columns)
    # Load saved model
    model = load('{}/{}.joblib'.format(config.get('model_dir'), config.get('model_file')))
    # Execute predictions
    LIP_SCORE, LIP = model.predict_proba(ds_predict)[:,1], model.predict(ds_predict)
    # Add columns to main DataFrame
    ds_residues['LIP_SCORE'] = LIP_SCORE
    ds_residues['LIP'] = LIP
    # Output results
    # If out_file flag is enabled, output predictions on file
    if config.get('out_file'):
        # Open output file
        with open(config.get('out_file'), 'w+') as of:
            # Write PDB id
            of.write('{}\n'.format(pdb_ids[0]))
            # Write residues info
            for idx, res in ds_residues.iterrows():
                of.write('{}/{}/{}//{} {} {}\n'.format(0, res['CHAIN_ID'], res['RES_ID'], res['RES_NAME'], np.round(res['LIP_SCORE'], 3), res['LIP']))
    # Otherwise, output predictions on terminal
    else:
        for idx, res in ds_residues.iterrows():
            print('{}/{}/{}//{} {} {}'.format(0, res['CHAIN_ID'], res['RES_ID'], res['RES_NAME'], np.round(res['LIP_SCORE'], 3), res['LIP']))


# TODO Define pipeline used for model training
def train_pipeline(config={}):
    # Get LIP/non-LIP file
    ds_training = pd.read_csv(config.get('lip_file'), sep='\t')
    # Extract PDB ids
    pdb_ids = set(ds_training.pdb.unique())
    # Remove excluded PDB ids
    if config.get('exclude'):
        pdb_ids -= set(config.get('exclude'))
    # Filter out invalid chains
    config['valid_chains'] = set([(row['pdb'], row['chain']) for idx, row in ds_training.iterrows()])
    # Define model
    model = None
    # Check if custom classifier must be used
    if config.get('model'):
        # Check if classifier is valid
        if config.get('model', {}).get('name') not in available_clf:
            logging.error('Selected model is not valid\nAborting...')
            exit()
        else:
            # Define model instance
            model = eval('{}'.format(config.get('model', {}).get('name')))
            # Build the model with given arguments
            model = model(**config.get('model', {}).get('args', {}))
    else:
        # Retrain stored model
        model = load('{}/{}.joblib'.format(config.get('model_dir'), config.get('model_file')))
    # Debug
    logging.debug('Model trained')
    logging.debug(model)
    # Extract features
    ds_residues, ds_predict = main_pipeline(pdb_ids, config)
    # Add LIP and LIP scores
    ds_residues = LIP_tag(ds_training, ds_residues)
    # Debug
    logging.debug('Datasets for training:')
    logging.debug(ds_predict.head())
    logging.debug(ds_residues.head())
    # Train model
    model.fit(ds_predict, ds_residues['LIP'])
    print('New model has been trained')
    # Overwrite the model
    dump(model, '{}/{}.joblib'.format(config.get('model_dir'), config.get('model_file')))
    print('New model has been saved to disk as {}/{}.joblib'.format(config.get('model_dir'), config.get('model_file')))
