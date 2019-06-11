# Import default libraries
import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
import re
from joblib import dump, load
# Import custom libraries
from modules.feature_extraction import *
from modules.feature_preprocessing import *

# Set debugging level (default DEBUG)
logging.basicConfig(level=logging.DEBUG)


# Initialize configuration dictionary
config = {
    # Default configuration file
    'config_file': './config.json'
}

# Load configuration settings from configuration file
with open(config.get('config_file')) as config_file:
    config = json.load(config_file)

# Debug config file
logging.debug(config)

# Parse arguments
parser = argparse.ArgumentParser(description='Predicts LIP flag for aminoacidic sequence defined by the given pdb object', prog='LIP_predictor')
# Define protein PDB ID
parser.add_argument('pdb_id', help='PDB id of the protein', type=str, action='store')
# Define window size
parser.add_argument('-ws', '--window_size', help='Size of the window used to average residues features', type=int, action='store')
# Define a force command do overwrite ring files
parser.add_argument('-rf', '--ring_force', help='Forces the program to download RING file relative to given PDB instance again', type=bool, action='store')
# Define ring files download directory
parser.add_argument('-rd', '--ring_dir', help='Folder where RING files will be downloaded', type=str, action='store')
# Define pdb files download directory
parser.add_argument('-pd', '--pdb_dir', help='Folder where PDB files will be downloaded', type=str, action='store')
# Define output file, if any
parser.add_argument('-of', '--out_file', help='Define in which file prediction results must be printed out', type=str, action='store')
# Define configuration file to overwrite momentaniously overwrite the default one
parser.add_argument('-cf', '--config_file', help='Define a custom configuration file, overwrites original parameters', type=str, action='store')
# Parse arguments
args = vars(parser.parse_args())
# Delete arguments which are None
args = {k: v for k, v in args.items() if v is not None}
# Debug
logging.debug(args)

# Check if there is a configuration file specified
new_config_file = args.get('config_file')
# Import config file
if new_config_file:
    # Open new configuration file
    with open(new_config_file) as new_config_file:
        # Get new configuration file content
        new_config = json.load(new_config_file)
        # Merge content into the default configuration file
        config = {**config, **new_config}

# Merge command line arguments into the config dictionary (they have highest priority)
config = {**config, **args}

# Debug
logging.debug(type(config))
logging.debug(config)

# Define a list of (1) PDB id
pdb_ids = list()
pdb_ids.append(config.get('pdb_id'))
# Retrieve PDB file
download_PDB(pdb_ids,  pdb_dir=config.get('pdb_dir'))
# Retrieve PDB data from PDB files
ds_residues = get_PDB(pdb_ids, pdb_dir=config.get('pdb_dir'))
# Compute Bag Of Words of RES_NAME feature
ds_residues = replace_bow(ds_residues,
                            col='RES_NAME',
                            vocabulary=config.get('vocabularies').get('res_name'),
                            prefix='RES_NAME_')
# Debug
logging.debug(ds_residues.head())

# Retrieve DSSP features
ds_dssp = get_DSSP(pdb_ids, pdb_dir=config.get('pdb_dir'), dssp_path=config.get('dssp_path'))
# Merge DSSP features into residues dataset
ds_residues = ds_residues.merge(ds_dssp, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
# Handle null values
ds_residues.fillna({'INTRA_CONTACTS': 0.0, 'INTER_CONTACTS': 0.0, 'INTRA_INTER_CONTACTS': 0.0}, inplace=True)
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

# Define a dataset for predictions
# Define the set of all columns
col = set(ds_residues.columns)
# Remove LIP and LIP_score columns
col -= set(['LIP', 'LIP_SCORE'])
# Delete INTRA_INTER_CONTACTS of the two columns INTRA_CONTACTS and INTER_CONTACTS
if not config.get('intra_inter_ratio'):
    col -= set(['INTRA_INTER_CONTACTS'])
else:
    col -= set(['INTRA_CONTACTS', 'INTER_CONTACTS'])
# Delete EDGE_LOC features
if config.get('no_edge_loc') or config.get('no_edge'):
    col -= set([c for c in col if re.search('^EDGE_LOC_*$', c)])
# Delete EDGE_TYPE features
if config.get('no_edge_type') or config.get('no_edge'):
    col -= set([c for c in col if re.search('^EDGE_TYPE_*$', c)])
# Update dataset for prediction (keeps columns in the correct order)
ds_residues = ds_residues.loc[:, [c for c in list(ds_residues.columns) if c in col]]
# Debug
logging.debug(ds_residues.head())

# TODO Execute sliding window averaging on main dataset

# Load saved model
model = load('{}/model.joblib'.format(config.get('model_dir')))
# Execute predictions
LIP_SCORE, LIP = model.predict_proba(ds_residues), model.predict(ds_residues)
# Add columns to main DataFrame
ds_residues['LIP_SCORE'] = LIP_SCORE
ds_residues['LIP'] = LIP
# Output results
# If out_file flag is enabled, output predictions on file
if config.get('out_file'):
    # Open output file
    with open(config.get('out_file')) as of:
        # Write PDB id
        of.write('{}\n'.format(pdb_ids[0]))
        # Write residues info
        for res in ds_residues.iterrows():
            out.write('{}/{}/{}//{} {} {}\n'.format(0, res.CHAIN_ID, res.RES_ID, res.LIP_SCORE, res.LIP))
# Otherwise, output predictions on terminal
else:
    for res in ds_residues.iterrows():
        print('{}/{}/{}//{} {} {}\n'.format(0, res.CHAIN_ID, res.RES_ID, res.LIP_SCORE, res.LIP))
