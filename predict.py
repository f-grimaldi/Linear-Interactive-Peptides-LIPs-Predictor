# Import default libraries
import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
from joblib import dump, load
# Import custom libraries
from modules.feature_extraction import *
from modules.feature_preprocessing import *

# Initialize configuration dictionary
config = {
    'config_file': './config.json',
    'debug_level': 'DEBUG'
}

# Load configuration settings from configuration file
with open(config.config_file) as config_file:
    config = json.load(config_file)

# Set debugging level (default DEBUG)
logging.basicConfig(level=logging[config.debug_level])
# Debug config file
logging.debug(config)

# Parse arguments
parser = argparse.ArgumentParser(description='Predicts LIP flag for aminoacidic sequence defined by the given pdb object', prog='LIP_predictor')
# Define protein PDB ID
parser.add_argument('pdb_id', '-pdb', '--pdb_id', help='PDB id of the protein', type=str, action='store')
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
args = parser.parse_args()

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
logging-debug(config)

# Define a list of (1) PDB id
pdb_ids = list(config.pdb_id)
# Retrieve PDB file
download_PDB(pdb_ids, pdb_dir=config.pdb_dir)
# Retrieve PDB data from PDB files
ds_residues = get_PDB(pdb_ids, pdb_dir=config.pdb_dir)
# Compute Bag Of Words of RES_NAME feature
ds_residues = replace_bow(ds_residues,
                            col='RES_NAME',
                            vocabulary=config.vocabularies.res_name,
                            prefix='RES_NAME_')

# Retrieve DSSP features
ds_dssp = get_DSSP(pdb_ids, pdb_dir=config.pdb_dir, dssp_path=config.dssp_path)
# Merge DSSP features into residues dataset
ds_residues = ds_residues.merge(ds_dssp, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
# Handle null values
ds_residues.fillna({'INTRA_CONTACTS': 0.0, 'INTER_CONTACTS': 0.0, 'INTRA_INTER_CONTACTS': 0.0}, inplace=True)

# Define empty list of files which will be donwloaded, identified by a pdb id
ring_downloads = []
# If forced download flag is enabled, set all ring files to be downloaded
if config.ring_force:
    ring_downloads = pdb_ids
else:
    # Define all files
    ring_files = ['{}/{}_network.zip'.format(config.ring_dir, pdb_id) for pdb_id in pdb_ids]
    # Append only PDB ids of files which have not already been downloaded
    ring_downloads = [pdb_id for pdb_id in pdb_ids if os.path.isfile('{}/{}_network.zip'.format(config.ring_dir, pdb_id))]
# Download RING files
download_RING(ring_downloads, ring_dir=config.ring_dir)

# Retrieve RING features
ds_ring = get_RING(pdb_ids, pdb_dir=config.pdb_dir, ring_dir=config.ring_dir, contact_threshold=config.contact_threshold)
# Compute edge features from RING dataset
ds_RING_edges = get_RING_edges(ds_RING)
# Merge edge features into residues dataset
ds_residues = ds_residues.merge(ds_RING_edges, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
# Handle null values
ds_residues.fillna({'EDGE_LOC': '', 'EDGE_TYPE': ''}, inplace=True)
# Compute Bag Of Words of EDGE_LOC feature
ds_residues = replace_bow(ds_residues,
                            col='EDGE_LOC',
                            vocabulary=config.vocabularies.edge_loc,
                            prefix='EDGE_LOC_')
# Compute Bag Of Words of EDGE_TYPE feature
ds_residues = replace_bow(ds_residues,
                            col='EDGE_TYPE',
                            vocabulary=config.vocabularies.edge_type,
                            prefix='EDGE_TYPE')

# Compute contact features from RING dataset
ds_RING_contacts = get_RING_contacts(ds_RING)
# Merge contact features into residues dataset
ds_residues = ds_residues.merge(ds_RING_edges, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
# Handle null values
ds_residues.fillna({'INTRA_CONTACTS': 0.0, 'INTER_CONTACTS': 0.0, 'INTRA_INTER_CONTACTS': 0.0}, inplace=True)

# TODO define a dataset for predictions

# TODO Execute sliding window averaging on main dataset

# Load saved model
model = load('{}/model.joblib'.format(config.model_dir))
# Execute predictions
LIP_SCORE, LIP = model.predict_proba(ds_residues),model.predict(ds_residues)
# Add columns to main DataFrame
ds_residues['LIP_SCORE'] = LIP_SCORE
ds_residues['LIP'] = LIP
# Output results
# If out_file flag is enabled, output predictions on file
if config.out_file:
    # Open output file
    with open(config.out_file) as of:
        # Write PDB id
        of.write('{}\n'.format(config.pdb_id))
        # Write residues info
        for res in ds_residues.iterrows():
            out.write('{}/{}/{}//{} {} {}'.format(0, res.CHAIN_ID, res.RES_ID, res.LIP_SCORE, res.LIP))
# Otherwise, output predictions on terminal
else:
    for res in ds_residues.iterrows():
