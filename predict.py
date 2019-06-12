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
from modules.pipelines import *

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
logging.debug(config)

# Define a list of (1) PDB id
pdb_ids = list()
pdb_ids.append(config.get('pdb_id'))

# Execute prediction pipeline
predict_pipeline(pdb_ids, config)
