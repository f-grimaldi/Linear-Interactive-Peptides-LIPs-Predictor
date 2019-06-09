# Import default libraries
import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
# Import custom libraries
from modules.feature_extraction import *

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
parser = argparse.ArgumentParser(description='Train a given model to predict LIP flag over aminoacidic sequences', prog='LIP learner')
# Define protein PDB ID
parser.add_argument('pdb_id', '-pdb', '--pdb_id', help='PDB id of the protein', type=str, action='store')
# Define random seed
parser.add_argument('-rs', --'random_seed', help='Random seed used in training', type=int, action='store')
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

# Retrieve DSSP features
ds_dssp = get_DSSP(pdb_ids, pdb_dir=config.pdb_dir, dssp_path=config.dssp_path)
# Merge DSSP features into residues dataset
ds_residues = ds_residues.merge(ds_dssp, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])

# Check if ring file has already been downloaded
# Download RING files
download_RING(pdb_ids, ring_dir=config.ring_dir)
# Retrieve RING features
ds_ring = get_RING(pdb_ids, pdb_dir=config.pdb_dir, ring_dir=config.ring_dir, contact_threshold=config.contact_threshold)
# Compute edge features from RING dataset
ds_RING_edges = get_RING_edges(ds_RING)
# Merge edge features into residues dataset
ds_residues = ds_residues.merge(ds_RING_edges, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
# Compute contact features from RING dataset
ds_RING_contacts = get_RING_contacts(ds_RING)
# Merge contact features into residues dataset
ds_residues = ds_residues.merge(ds_RING_edges, how='left', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
# Move LIP_SCORE and LIP columns to the latest positions
ds_residues = ds_residues.drop(columns=['LIP_SCORE', 'LIP']).assign(LIP_SCORE=ds_residues['LIP_SCORE'], LIP=ds_residues['LIP'])

# Test on saved model


# Merge features from dssp and ring into
