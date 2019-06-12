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
from scipy import signal
from sklearn.feature_extraction.text import CountVectorizer
# Import BioPython utils
from Bio.PDB import PDBList, calc_angle, calc_dihedral, PPBuilder, is_aa, PDBIO, NeighborSearch, DSSP, HSExposureCB
from Bio.PDB.PDBParser import PDBParser


# Function for tagging residues as LIP/non-LIP
# Overwrites entries in the second dataset with LIP flag accrodingly to the first dataset
def LIP_tag(ds_original, ds_residues):
    #For every protein we take the information of where LIP residue are
    for idx, row in ds_original.iterrows():
        # Bind information to correct variables
        pdb, chain, start, end = (row[0:4])
        # Skips if start and and are 'neg'
        if (start == 'neg') or (end == 'neg'):
            continue
        # Cast start and end values to integer
        start, end = int(start), int(end)
        # Get the correct slice of data which will be edited
        sliced = ((ds_residues['PDB_ID'] == pdb)
                    & (ds_residues['CHAIN_ID'] == chain)
                    & (ds_residues['RES_ID'] <= end)
                    & (ds_residues['RES_ID'] >= start))
        #Now we set to 1 all the residue whose features are the one desired
        ds_residues.loc[sliced, 'LIP'] = 1
        ds_residues.loc[sliced, 'LIP_SCORE'] = 1
    return ds_residues


# Download PDB specified in a list of PDB ids
def download_PDB(pdb_ids, pdb_dir='.'):
    # Define pdb file fetching class
    pdbl = PDBList()
    # Fetch every protein
    for pdb_id in pdb_ids:
        # Debug
        logging.debug('PDB file which will be downloaded')
        logging.debug(pdb_id)
        # Execute fetching of the protein (pdb file)
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')

# Function for getting residues contained into PDB files, given their ids
def get_PDB(pdb_ids, valid_chains=None, chain_len=True, pdb_dir='.'):
    # Debug
    logging.debug('Directory for PDB files')
    logging.debug(pdb_dir)
    logging.debug('Chain length')
    logging.debug(chain_len)
    logging.debug('Valid chains')
    logging.debug(valid_chains)
    # New list for residues
    # It will be turned into DataFrame later
    ds_residues = list()
    # Loop thorugh every protein
    for pdb_id in pdb_ids:
        # Define an array of aminoacids for the current protein
        residues = list()
        # Get structure of the protein
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_dir + '/pdb{}.ent'.format(pdb_id))
        # We select only the 0-th model
        model = structure[0]
        # Loop through every model's chain
        for chain in model:
            # Check chain is in valid chains
            if (valid_chains is not None) and ((pdb_id, chain.id) not in valid_chains):
                continue
            for residue in chain:
                # Do not take into account non-aminoacidic residues (e.g. water molecules)
                if not is_aa(residue):
                    continue
                # Add an entry to the residues list
                residues.append((pdb_id, model.id, chain.id, residue.id[1], residue.get_resname(), 0, 0))
        if not residues:
            logging.warning('A protein {} has no valid residues'.format(pdb_id))
        ds_residues += residues
    if not ds_residues:
        logging.error('No valid aminoacidics found\nAborting...')
        exit()
    # Turn list into dataframe
    ds_residues = pd.DataFrame(ds_residues)
    # Debug
    logging.debug('PDB dataset')
    logging.debug(ds_residues)
    # Define dataset column names
    ds_residues.columns = ['PDB_ID', 'MODEL_ID', 'CHAIN_ID', 'RES_ID', 'RES_NAME', 'LIP_SCORE', 'LIP']
    # Check if chain lengths should be added
    if chain_len:
        # Group and extract chain length
        ds_chain_len = ds_residues.groupby(['PDB_ID', 'MODEL_ID', 'CHAIN_ID']).size().reset_index(name='CHAIN_LEN')
        # Add chain len to main dataframe
        ds_residues = ds_residues.merge(ds_chain_len, how='left', on=['PDB_ID', 'MODEL_ID', 'CHAIN_ID'])
        # Reindex columns of the main dataframe: chain length after chain id
        ds_residues = ds_residues.reindex(['PDB_ID', 'MODEL_ID', 'CHAIN_ID', 'CHAIN_LEN', 'RES_ID', 'RES_NAME', 'LIP_SCORE', 'LIP'], axis=1)
    # Show some info about the dataset
    logging.debug("Numbers of proteins: {}".format(len(pdb_ids)))
    logging.debug("Numbers of residues: {}".format(len(ds_residues.PDB_ID)))
    # Return created dataset
    return ds_residues


# Function for extracting DSSP data from a list of pdb_ids
def get_DSSP(pdb_ids, pdb_dir='.', dssp_path='/usr/local/bin/mkdssp', drop_features=['DSSP_ID', 'AA']):
    # Check parameters
    logging.debug('PDB ids: {}'.format(pdb_ids))
    logging.debug('PDB directory: \'{}\''.format(pdb_dir))
    # Define a list of dssp features (which will be stored in a list before being turned into DataFrame
    ds_dssp = list()
    # Loop thorugh every protein
    for pdb_id in pdb_ids:
        # Parse structure of the protein
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_dir + '/pdb{}.ent'.format(pdb_id))
        # Get only first model
        model = structure[0]
        # Define DSSP instance of the 0-th model
        dssp = DSSP(model, pdb_dir + '/pdb{}.ent'.format(pdb_id), dssp="/usr/local/bin/mkdssp")
        # Get DSSP features: dssp index, amino acid, secondary structure, relative ASA, phi, psi, NH_O_1_relidx,
        # NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy
        # Get chain id and residue id
        for ids, res in zip(dict(dssp), dssp):
            # Create the DSSP row
            row = list()
            row.append(pdb_id)
            row.extend(list(ids))
            row.extend(list(res))
            # Add row to dssp list
            ds_dssp.append(row)
    # Define feature names
    columns = ['PDB_ID', 'CHAIN_ID', 'RES_ID', 'DSSP_ID', 'AA', 'SEC_STRUCT', 'REL_ASA', 'PHI', 'PSI',
               'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx', 'O_NH_1_energy', 'NH_O_2_relidx',
               'H_O_2_energy', 'O_NH_2_relidx', 'O_NH_2_energy']
    # Define DSSP DataFrame
    ds_dssp = pd.DataFrame(ds_dssp, columns=columns)
    # Turn RES_ID from tuple to integer (gets 1-st element)
    ds_dssp.RES_ID = ds_dssp.RES_ID.apply(lambda x: x[1])
    # Drop useless features, if any
    if drop_features:
        ds_dssp = ds_dssp.drop(drop_features, axis=1)
    # Turns NA to nan
    ds_dssp = ds_dssp.replace('NA', np.nan)
    # Handle nan
    ds_dssp.loc[ds_dssp.REL_ASA.isna(), 'REL_ASA'] = ds_dssp.REL_ASA.mean()
    # Return DSSP dataset
    return ds_dssp


# Function for dwonloading RING archives features from one or more pdb files/ids
def download_RING(pdb_ids, ring_dir='.', time_sleep=5):
    # Initialize the status of every RING server job into a dict pdb_id -> status
    job_status = ['nd' for pdb_id in pdb_ids]
    # Initialize job_ids returned from RING server
    job_ids = ['nd' for pdb_id in pdb_ids]
    # Loop through every protein
    for i in range(0, len(pdb_ids)):
        # Define POST request parameters to tell RING web service to start elaborating
        req = {"pdbName": pdb_ids[i], "chain": "all", "seqSeparation": "5", "networkPolicy": "closest",
               "nowater": "true", "ringmd": "false", "allEdges": "true",
               "thresholds": '{"hbond": 3.5, "vdw": 0.5, "ionic": 4, "pipi": 6.5, "pication": 5, "disulphide": 2.5}'}
        # Define a request object to handle connection
        r = requests.post('http://protein.bio.unipd.it/ringws/submit',
                          data=json.dumps(req),
                          headers={'content-type': 'application/json'})
        # TODO error on RING request
        if r.status_code != 200:
            logging.error('Server responded with error\nAborting...')
            exit()
        # Debug
        logging.debug('RING request')
        logging.debug(req)
        logging.debug('RING output file')
        logging.debug(r.text)
        # Define id of the job provided by RING
        job_ids[i] = json.loads(r.text)['jobid']
    # Debug info
    logging.debug(zip(pdb_ids, job_ids))
    # Repeat until every job is complete
    while any([(js != 'complete') for js in job_status]):
        # Loop through every request to check its status
        for i in range(0, len(pdb_ids)):
            # Proceed only if not already completed
            if job_status[i] != 'complete':
                # Retrieve (update) current server elaboration status for this specific job
                job_status[i] = json.loads(
                    requests.get('http://protein.bio.unipd.it/ringws/status/{}'.format(job_ids[i])).text).get("status", None)
                # Skip if already downloaded
                if job_status[i] == 'complete':
                    # Define archive file name
                    archive_file = "{}_network.zip".format(pdb_ids[i])
                    # Download RING output as an archive
                    r = requests.get("http://protein.bio.unipd.it/ring_download/{}/{}".format(job_ids[i], archive_file))
                    # Write the archive file
                    with open(ring_dir + '/' + archive_file, "wb") as fout:
                        fout.write(r.content)
            # Debug info
            logging.debug("Status for pdb {}: {} with job id {}".format(pdb_ids[i], job_status[i], job_ids[i]))
        # Wait for server elaboration
        time.sleep(time_sleep)


# Creates a DataFrame of RING features only
def get_RING(pdb_ids, pdb_dir='.', ring_dir='.', contact_threshold=3.5):
    # Define a list of RING features
    ring_list = list()
    # Loop through every PDB ID passed as argument
    for pdb_id in pdb_ids:
        # Parse structure
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_dir + '/' + 'pdb{}.ent'.format(pdb_id))
        # Create chains with residue ids compatible with RING
        nodes = {}
        for chain in structure[0]:
            nodes[chain.id] = []
            for residue in chain:
                if residue.id[0] == ' ':
                    node = "{}:{}:{}:{}".format(chain.id,
                                                residue.id[1],
                                                residue.id[2] if residue.id[2] != ' ' else '_',
                                                residue.get_resname())
                    nodes[chain.id].append(node)
        # Opens ZIP file
        zf = zipfile.ZipFile(ring_dir + '/' + '{}_network.zip'.format(pdb_id), 'r')
        # Opens edges file
        with zf.open('{}_edges.txt'.format(pdb_id), 'r') as f:
            # Skip first row: header
            next(f)
            # Reads edges line by line
            for line in f:
                # Encodes string as utf8 from bytes
                line = line.decode('utf-8')
                # node = chain:residue_number:insertion_code:residue_name
                # edge = localization:contact_type (localization MC = main chain, SC = side chain)
                node_a, edge, node_b, distance, _, _, atom_a, atom_b = line.split(sep='\t')[0:8]
                distance = float(distance)
                if distance <= contact_threshold:
                    node_a = node_a.split(':')
                    node_b = node_b.split(':')
                    edge_type, edge_loc = edge.split(':')
                    # Create a contact instance: (PDB_ID, CHAIN_ID(A), RESIDUE_ID(A), CHAIN_ID(B), RESIDUE_ID(B), other RING features)
                    ring_list.append((pdb_id, node_a[0], node_a[1], node_b[0], node_b[1], edge_loc, atom_b, atom_a, edge_type))
                    ring_list.append((pdb_id, node_b[0], node_b[1], node_a[0], node_a[1], edge_loc, atom_a, atom_b, edge_type))
        # Define RING features columns
        columns = ['PDB_ID', 'CHAIN_ID', 'RES_ID', 'CHAIN_ID_B', 'RES_ID_B', 'EDGE_LOC', 'ATOM_A', 'ATOM_B', 'EDGE_TYPE']
        # Turn RING features list into DataFrame
        ds_ring = pd.DataFrame(ring_list, columns=columns)
        # Turn residue id into integer
        ds_ring.RES_ID = ds_ring.RES_ID.astype(int)
        ds_ring.RES_ID_B = ds_ring.RES_ID_B.astype(int)
        # Return RING features
        return ds_ring


# Get features relative to edges, grouped by residue
def get_RING_edges(ds_ring, drop_features=[]):
    # Groups edge attributes per residue
    group_by = ds_ring.groupby(['PDB_ID', 'CHAIN_ID', 'RES_ID'])
    ds_edges = group_by['EDGE_LOC'].apply(lambda x: ' '.join(x)).reset_index()
    ds_edges['EDGE_TYPE'] = group_by['EDGE_TYPE'].apply(lambda x: ' '.join(x)).reset_index()['EDGE_TYPE']
    return ds_edges
    # Remove undesidered features
    if drop_features:
        ds_edges = ds_edges.drop(drop_features, axis=1)
    # Return dataset with RING edges info per residue
    return ds_edges


# Get features relative to contacts, grouped by residue
def get_RING_contacts(ds_ring, drop_features=[]):
    # Define a slice of the ring DataFrame
    ds_ring = ds_ring[['PDB_ID', 'CHAIN_ID', 'RES_ID', 'CHAIN_ID_B', 'RES_ID_B']]
    # Define dataset of intra-chain contacts
    ds_intra = ds_ring[ds_ring.CHAIN_ID == ds_ring.CHAIN_ID_B]
    ds_inter = ds_ring[ds_ring.CHAIN_ID != ds_ring.CHAIN_ID_B]
    # Create contacts dataset with intra chain contacts
    ds_intra = (ds_intra.groupby(['PDB_ID', 'CHAIN_ID', 'RES_ID'], as_index=False)
                        .size()
                        .reset_index(name='INTRA_CONTACTS'))
    # Add compute inter-chain contacts
    ds_inter = (ds_inter.groupby(['PDB_ID', 'CHAIN_ID', 'RES_ID'], as_index=False)
                        .size()
                        .reset_index(name='INTER_CONTACTS'))
    # Merges computed inter-contacts into the returned DataFrame
    ds_contacts = pd.merge(ds_intra, ds_inter, how='outer', on=['PDB_ID', 'CHAIN_ID', 'RES_ID'])
    # Handle NaNs
    ds_contacts.fillna({'INTRA_CONTACTS':0, 'INTER_CONTACTS':0}, inplace=True)
    # Compute intra/inter ratio
    num = np.array(ds_contacts['INTRA_CONTACTS'])
    den = np.array(ds_contacts['INTER_CONTACTS']) + 0.1
    ds_contacts['INTRA_INTER_CONTACTS'] = num / den
    # Remove undesidered features
    if drop_features:
        ds_contacts = ds_contacts.drop(drop_features, axis=1)
    # Return dataset with RING edges info per residue
    return ds_contacts
